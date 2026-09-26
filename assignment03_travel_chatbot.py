"""Assignment 3: Strands Travel Planner Chatbot

A Gradio chatbot powered by Strands Agents that plans a one-day trip to any city.

For each city the agent:
  1. gets the current weather           (OpenWeather)
  2. finds 3 popular attractions         (Tavily web search)
  3. calculates the total entry cost     (calculator tool, plain Python)
  4. replies with a one-day itinerary    (in the Gradio chat window)

"""

import os

import gradio as gr
import requests
from dotenv import load_dotenv
from strands import Agent, tool
from strands.models.openai import OpenAIModel
from strands.tools.executors import SequentialToolExecutor
from tavily import TavilyClient

load_dotenv()

REQUIRED_KEYS = ["GROQ_API_KEY", "OPENWEATHER_API_KEY", "TAVILY_API_KEY"]
missing_keys = [name for name in REQUIRED_KEYS if not os.getenv(name)]
if missing_keys:
    raise SystemExit(f"Missing in .env: {', '.join(missing_keys)}")

OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
REQUEST_TIMEOUT_SECONDS = 10
MAX_SEARCH_RESULTS = 3
MAX_CONTENT_CHARS = 600      # keep search results short to save tokens
NUMBER_OF_ATTRACTIONS = 3


tool_calls_log = []
last_calculation = {}


@tool
def get_current_weather(city: str) -> dict:
    """Gets the current weather for a city from the OpenWeather API (metric units, °C).

    Args:
        city: The city name, optionally with a country code, e.g. "Kathmandu" or "Kathmandu,NP".
    """
    print(f"[TOOL EXECUTED] get_current_weather(city='{city}')")
    tool_calls_log.append("Weather")

    def failure(message: str) -> dict:
        print(f"    → ERROR: {message}")
        return {"ok": False, "city": city, "error": message}

    params = {"q": city, "appid": os.getenv("OPENWEATHER_API_KEY"), "units": "metric"}

    # SECURITY: the request URL contains the API key, so we never show the raw
    # exception text or response.url — only our own short messages.
    try:
        response = requests.get(OPENWEATHER_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.Timeout:
        return failure("OpenWeather did not respond in time.")
    except requests.exceptions.RequestException:
        return failure("Could not connect to OpenWeather.")

    if response.status_code == 401:
        return failure("OpenWeather rejected the API key (HTTP 401).")
    if response.status_code == 404:
        return failure(f"City not found by OpenWeather: '{city}'.")
    if response.status_code != 200:
        return failure(f"OpenWeather returned HTTP {response.status_code}.")

    try:
        data = response.json()
        result = {
            "ok": True,
            "city": data["name"],
            "country": data.get("sys", {}).get("country", ""),
            "temperature_c": round(float(data["main"]["temp"]), 1),
            "feels_like_c": round(float(data["main"]["feels_like"]), 1),
            "description": data["weather"][0]["description"],
        }
    except (ValueError, KeyError, IndexError, TypeError):
        return failure("OpenWeather's response was missing expected fields.")

    print(f"    → {result['city']}, {result['country']}: {result['temperature_c']} °C, {result['description']}")
    return result


tavily_client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


@tool
def search_web(query: str) -> dict:
    """Performs a web search using Tavily to get real-time information,
    such as popular attractions in a city or an attraction's entry fee.

    Args:
        query: The search query string, e.g. "top 3 tourist attractions in Kathmandu".
    """
    print(f"[TOOL EXECUTED] search_web(query='{query}')")
    tool_calls_log.append("Search")

    try:
        response = tavily_client.search(query=query, search_depth="basic", max_results=MAX_SEARCH_RESULTS)
    except Exception as exc:  # e.g. network problem, bad key, usage limit
        message = f"Web search failed ({type(exc).__name__})."
        print(f"    → ERROR: {message}")
        return {"ok": False, "query": query, "error": message}

    results = [
        {"title": r.get("title"), "content": (r.get("content") or "")[:MAX_CONTENT_CHARS], "url": r.get("url")}
        for r in response.get("results", [])
    ]
    print(f"    → {len(results)} result(s)")
    return {"ok": True, "query": query, "results": results}


@tool
def calculate_total_cost(attractions: list[str], costs: list[float], currency: str) -> dict:
    """Adds up the admission costs of exactly 3 attractions and returns the exact total.
    Always use this tool for the total — do not add the numbers yourself.

    Args:
        attractions: The names of the 3 attractions, in the same order as `costs`.
        costs: The admission cost of each attraction, all in the same currency (use 0 for free entry).
        currency: The currency code of the costs, e.g. "NPR" or "USD".
    """
    print(f"[TOOL EXECUTED] calculate_total_cost(attractions={attractions}, costs={costs}, currency='{currency}')")
    tool_calls_log.append("Calculator")

    def failure(message: str) -> dict:
        print(f"    → ERROR: {message}")
        return {"ok": False, "error": message}

    if len(attractions) != NUMBER_OF_ATTRACTIONS:
        return failure(f"Expected exactly {NUMBER_OF_ATTRACTIONS} attractions, got {len(attractions)}.")
    if len(costs) != len(attractions):
        return failure("Each attraction needs exactly one cost (same order as the names).")
    if any(cost < 0 for cost in costs):
        return failure("Costs cannot be negative.")

    total = round(sum(costs), 2)
    calculation = " + ".join(f"{name} ({cost:g})" for name, cost in zip(attractions, costs)) + f" = {total:g} {currency}"

    last_calculation.clear()
    last_calculation.update({"total": total, "currency": currency, "calculation": calculation})

    print(f"    → {calculation}")
    return {"ok": True, "total": total, "currency": currency, "calculation": calculation}


model = OpenAIModel(
    model_id="openai/gpt-oss-120b",
    client_args={
        "api_key": os.getenv("GROQ_API_KEY"),
        "base_url": "https://api.groq.com/openai/v1",
    },
)


SYSTEM_PROMPT = """You are a friendly travel planner chatbot that creates a concise one-day travel plan for a city.

If the user's message doesn't name a city, reply briefly asking which city they'd like to visit,
and do not call any tools.

Otherwise, follow these steps using the tools — never invent weather, attractions, prices or totals:
1. Call get_current_weather for the city.
2. Call search_web to find 3 popular tourist attractions in the city.
3. Call search_web to find the admission / entry fee of each attraction.
   - Use prices exactly as found. Keep all costs in ONE currency (prefer the local currency).
   - If an attraction is free, use 0. If no price can be found, use 0 and mark it "price not found".
4. Call calculate_total_cost with the 3 attraction names and their costs (in the same order).
   Report the calculation and total EXACTLY as the tool returns them.
5. If the weather tool says the city was not found, tell the user and ask them to check the spelling.

Write the final plan in Markdown, in this format:

### One-Day Travel Plan: <City>

**Weather (weather tool):** <temperature> °C, <description>

**Top 3 Attractions (web search):**
1. **<Name>** — admission: <price or "free" or "price not found"> (source: <website name>)
2. ...
3. ...

**Estimated Attraction Cost (calculator tool):** <calculation from the tool>

**Itinerary:**
- **08:00** — ...
- (5–6 short lines visiting the 3 attractions, with a lunch break and weather-appropriate advice)

**Notes:** <one short line on assumptions, e.g. prices are for foreign visitors and may change>
"""


def create_travel_agent() -> Agent:
    """Creates a fresh agent for each message, so one city's conversation doesn't affect the next."""
    return Agent(
        model=model,
        tools=[get_current_weather, search_web, calculate_total_cost],
        tool_executor=SequentialToolExecutor(),   # run tool calls one at a time
        system_prompt=SYSTEM_PROMPT,
        callback_handler=None,                    # don't print streamed text in the terminal
    )


def tools_summary() -> str:
    """A short footer showing which tools the agent used and the calculator's total."""
    if not tool_calls_log:
        return ""
    counts = {name: tool_calls_log.count(name) for name in ["Weather", "Search", "Calculator"]}
    used = " · ".join(f"{name} ×{count}" for name, count in counts.items() if count)
    footer = f"\n\n---\n🔧 **Tools used:** {used}"
    if last_calculation:
        footer += f"  \n🧮 **Calculator result:** {last_calculation['calculation']}"
    elif counts["Search"]:   # a plan was attempted but no total was calculated
        footer += "  \n⚠️ The calculator tool wasn't used, so there is no verified total."
    return footer


def chat(message, history):
    """Called by Gradio for every user message. Returns the bot's reply (Markdown)."""
    message = (message or "").strip()
    if not message:
        return "Please type a city name, for example **Kathmandu** or **Plan a day in Pokhara**."

    tool_calls_log.clear()
    last_calculation.clear()
    print(f"\n=== New request: {message!r} ===")

    try:
        response = create_travel_agent()(message)
        reply = str(response).strip()
    except Exception as exc:   # e.g. Groq rate limit or network problem
        print(f"    → Agent error: {type(exc).__name__}")
        return ("Sorry, I couldn't finish the plan right now "
                f"({type(exc).__name__}). Please wait a moment and try again.")

    return (reply or "Sorry, I didn't get a response. Please try again.") + tools_summary()


demo = gr.ChatInterface(
    fn=chat,
    title="✈️ Strands Travel Planner",
    description=(
        "Type a city and get a one-day travel plan: current weather, 3 popular attractions, "
        "their total entry cost, and an itinerary. Powered by Strands + Groq (gpt-oss-120b). "
        "Planning takes about 20–40 seconds."
    ),
    examples=["Kathmandu", "Plan a day in Pokhara", "Tokyo"],
)

if __name__ == "__main__":
    demo.launch() 
