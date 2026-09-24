"""🌤️ Weather Agent — a localhost chatbot for Assignment 1.

Type a request such as "Get the current weather for Kathmandu, London and Tokyo".

How it works (same ideas as Assignment01_Weather_Agent.ipynb):
  1. Gemini reads your prompt and identifies the locations (structured JSON
     output with Pydantic — notebook Level 3). Up to 3 locations per request.
  2. Gemini then fetches the weather with the get_current_weather tool using
     MANUAL function calling (Level 6), one location per turn (Level 7).
  3. Python runs every OpenWeather request itself, one after another.
  4. Python — not Gemini — calculates the average temperature, and only when
     every requested location returned a valid reading.

Failover: Gemini is the primary LLM. If Gemini returns a rate-limit/quota
error (HTTP 429 / RESOURCE_EXHAUSTED), the same steps are handed to Groq so
the chatbot keeps working. Any other Gemini error is shown, not failed over.

Start it from the project folder with:
    .venv/bin/python -m streamlit run weather_agent_ui.py

To test the failover without using up your real Gemini quota:
    SIMULATE_GEMINI_RATE_LIMIT=1 .venv/bin/python -m streamlit run weather_agent_ui.py

API keys are read from the project's .env file and are never displayed.
"""

import json
import os
from pathlib import Path
from typing import Callable, Optional

import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError

# Groq is optional: the app still works with Gemini alone if it isn't installed.
try:
    import groq
except ImportError:
    groq = None

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Load .env from the folder this file lives in, so it works from any directory.
load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL_ID = "gemini-3.6-flash"              # primary provider
# Fallback provider. The app uses the first of these that your Groq key can access
# (all support tool calling). See choose_groq_model().
GROQ_MODEL_CANDIDATES = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b",
                         "openai/gpt-oss-20b", "llama-3.1-8b-instant"]
MAX_LOCATIONS = 3              # Assignment 1 is based on three locations
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
UNITS = "metric"               # "metric" -> temperatures in degrees Celsius
REQUEST_TIMEOUT_SECONDS = 10   # give up on a slow OpenWeather request
MAX_AGENT_TURNS = 6            # safety limit so the agent loop can't run forever
REQUIRED_KEYS = ["GEMINI_API_KEY", "OPENWEATHER_API_KEY"]   # GROQ_API_KEY is optional (backup only)

# Test switch, OFF unless you start the app with SIMULATE_GEMINI_RATE_LIMIT=1.
# When on, Gemini's first call raises a realistic 429 error so you can watch the fallback.
SIMULATE_GEMINI_RATE_LIMIT = os.getenv("SIMULATE_GEMINI_RATE_LIMIT") == "1"


# ---------------------------------------------------------------------------
# The OpenWeather tool (same as notebook Cell C)
# ---------------------------------------------------------------------------
def get_current_weather(city: str) -> dict:
    """Gets the current weather for one location from the OpenWeather API.

    Call this tool once per location. Temperatures are in degrees Celsius.

    Args:
        city: The location as "City,CountryCode", for example "Kathmandu,NP".
    """
    city = (city or "").strip()

    def failure(message: str) -> dict:
        return {"ok": False, "requested_location": city, "error": message}

    if not city:
        return failure("No location was given.")

    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return failure("OPENWEATHER_API_KEY is not set.")

    params = {"q": city, "appid": api_key, "units": UNITS}

    # SECURITY: the request URL contains the API key (appid=...), and requests'
    # exception messages can include that URL. So we never show the raw
    # exception text or response.url — only our own fixed messages.
    try:
        response = requests.get(OPENWEATHER_URL, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
    except requests.exceptions.Timeout:
        return failure(f"OpenWeather did not respond within {REQUEST_TIMEOUT_SECONDS} seconds.")
    except requests.exceptions.ConnectionError:
        return failure("Could not connect to OpenWeather (check your internet connection).")
    except requests.exceptions.RequestException:
        return failure("The request to OpenWeather failed.")

    if response.status_code == 401:
        return failure("OpenWeather rejected the API key (HTTP 401).")
    if response.status_code == 404:
        return failure(f"Location not found by OpenWeather: '{city}' (HTTP 404).")
    if response.status_code == 429:
        return failure("OpenWeather rate limit reached (HTTP 429). Try again shortly.")
    if response.status_code != 200:
        return failure(f"OpenWeather returned an unexpected status (HTTP {response.status_code}).")

    try:
        data = response.json()
    except ValueError:
        return failure("OpenWeather returned a response that is not valid JSON.")

    try:
        temperature = data["main"]["temp"]
        city_name = data["name"]
        country = data.get("sys", {}).get("country", "")
        description = data["weather"][0]["description"]
    except (KeyError, IndexError, TypeError, AttributeError):
        return failure("OpenWeather's response was missing expected weather fields.")

    # bool is a subclass of int in Python, so rule it out explicitly.
    if isinstance(temperature, bool) or not isinstance(temperature, (int, float)):
        return failure("OpenWeather returned a temperature that is not a number.")

    return {
        "ok": True,
        "requested_location": city,
        "city": city_name,
        "country": country,
        "temperature_c": float(temperature),
        "description": description,
    }


# ---------------------------------------------------------------------------
# Safe Gemini error messages (never show raw exception text)
# ---------------------------------------------------------------------------
def safe_gemini_error(exc: Exception) -> str:
    """Turns a Gemini SDK exception into a short, safe message."""
    code = getattr(exc, "code", None)
    key_problem = code == 400 and "api key" in str(getattr(exc, "message", "") or "").lower()
    if code in (401, 403) or key_problem:
        return "Gemini rejected the API key or permissions. Check GEMINI_API_KEY in .env."
    if code == 404:
        return f"Gemini model '{MODEL_ID}' was not found."
    if code == 429:
        return "Gemini rate limit reached. Please wait a moment and try again."
    if isinstance(code, int) and code >= 500:
        return "Gemini is temporarily unavailable. Please try again."
    if isinstance(exc, errors.APIError):
        return f"Gemini could not process the request (error {code})."
    return f"Could not reach Gemini ({type(exc).__name__})."


def is_rate_limit_error(exc: Exception) -> bool:
    """True only for Gemini rate-limit / quota errors — the ONLY errors that trigger Groq.

    Other errors (bad API key, bad request, network, bugs in our code) return
    False, so they are reported to the user instead of being silently failed over.
    """
    if not isinstance(exc, errors.APIError):
        return False
    if exc.code == 429:                                   # HTTP 429 Too Many Requests
        return True
    if str(exc.status or "").upper() == "RESOURCE_EXHAUSTED":
        return True
    message = str(exc.message or "").lower()              # only inspected, never displayed
    return "quota" in message or "rate limit" in message


def safe_groq_error(exc: Exception, groq_model: str = "") -> str:
    """Turns a Groq SDK exception into a short, safe message (Groq is only used as the backup)."""
    if groq is not None:
        if isinstance(exc, groq.RateLimitError):
            return ("Both providers are rate-limited right now: Gemini hit its limit and the Groq "
                    "backup did too. Please wait a minute and try again.")
        if isinstance(exc, (groq.AuthenticationError, groq.PermissionDeniedError)):
            return "Gemini is rate-limited and the Groq backup rejected its key. Check GROQ_API_KEY in .env."
        if isinstance(exc, groq.NotFoundError):
            return (f"Gemini is rate-limited and the Groq model '{groq_model}' is not available "
                    "to your Groq key. Check which models your key can use in the Groq console.")
        if isinstance(exc, (groq.APIConnectionError, groq.APITimeoutError)):
            return "Gemini is rate-limited and the Groq backup could not be reached."
        if isinstance(exc, groq.APIStatusError):
            return f"Gemini is rate-limited and the Groq backup returned an error ({exc.status_code})."
    return f"Gemini is rate-limited and the Groq backup failed ({type(exc).__name__})."


# ---------------------------------------------------------------------------
# Step 1: Gemini identifies the locations (structured output, notebook Level 3)
# ---------------------------------------------------------------------------
class LocationRequest(BaseModel):
    is_weather_request: bool = Field(
        description="True if the user is asking for the current weather of one or more places.")
    locations: list[str] = Field(
        description='Every place the user asked about, in the order mentioned, formatted as '
                    '"City,CountryCode" (e.g. "Kathmandu,NP"). Use just "City" if the country is unclear.')


EXTRACTION_INSTRUCTION = (
    "You extract locations from a user's weather request. List every place the user "
    "asked about, in the order they were mentioned, with no duplicates. Add the ISO "
    "3166 two-letter country code only when you are confident of it. Do not add places "
    "the user did not mention, and do not answer the question yourself."
)

EXTRACTION_CONFIG = types.GenerateContentConfig(
    system_instruction=EXTRACTION_INSTRUCTION,
    response_mime_type="application/json",
    response_schema=LocationRequest,
    temperature=0.0,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def gemini_extract_locations(client: genai.Client, prompt: str) -> tuple[Optional[LocationRequest], Optional[str]]:
    """Asks Gemini which locations the prompt mentions. Returns (request, error_message).

    Gemini API errors are NOT caught here: they go up to run_with_failover(),
    which decides whether it's a rate limit (-> try Groq) or a real error.
    """
    if SIMULATE_GEMINI_RATE_LIMIT:
        # Test mode only: raise the same kind of error Gemini sends when its quota runs out.
        raise errors.APIError(429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED",
                                              "message": "Simulated quota exceeded (test mode)."}})

    response = client.models.generate_content(model=MODEL_ID, contents=prompt, config=EXTRACTION_CONFIG)

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, LocationRequest):
        return parsed, None
    try:
        return LocationRequest.model_validate_json(response.text or ""), None
    except (ValidationError, ValueError):
        return None, "Gemini did not return a readable list of locations. Please rephrase your request."


def groq_extract_locations(groq_client, groq_model: str, prompt: str) -> tuple[Optional[LocationRequest], Optional[str]]:
    """Groq version of the same step, using Groq's JSON mode. Never raises."""
    try:
        response = groq_client.chat.completions.create(
            model=groq_model,
            messages=[
                {"role": "system", "content": EXTRACTION_INSTRUCTION + (
                    ' Reply with JSON only, in this shape: '
                    '{"is_weather_request": true, "locations": ["City,CountryCode"]}')},
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.0,
        )
    except Exception as exc:
        return None, safe_groq_error(exc, groq_model)
    try:
        return LocationRequest.model_validate_json(response.choices[0].message.content or ""), None
    except (ValidationError, ValueError, IndexError):
        return None, "Groq did not return a readable list of locations. Please rephrase your request."


def _city_key(location: str) -> str:
    """'Kathmandu,NP' -> 'kathmandu'. Used to match locations to readings."""
    return (location or "").split(",")[0].strip().lower()


def unique_locations(locations: list[str]) -> list[str]:
    """Removes blanks and duplicate city names, keeping the original order."""
    seen, result = set(), []
    for location in locations:
        location = (location or "").strip()
        if location and _city_key(location) not in seen:
            seen.add(_city_key(location))
            result.append(location)
    return result


# ---------------------------------------------------------------------------
# Step 2: the agent loop (notebook Cell E — manual function calling)
# ---------------------------------------------------------------------------
AGENT_SYSTEM_INSTRUCTION = (
    "You are a weather data agent with one tool: get_current_weather.\n"
    "Rules:\n"
    "1. Always use get_current_weather for current weather. Never invent or estimate weather data.\n"
    "2. Call the tool exactly once for each listed location.\n"
    "3. Work sequentially: request only ONE location per turn, in the order given, "
    "and wait for its result before requesting the next.\n"
    "4. Pass each location string exactly as listed (for example 'Kathmandu,NP').\n"
    "5. If a tool result has ok=false, report that location's error clearly.\n"
    "6. Do not calculate an average or any other number yourself; the program "
    "calculates the authoritative average from the tool results.\n"
    "7. When every location is done, give a short, friendly summary of each location's weather."
)

AGENT_CONFIG = types.GenerateContentConfig(
    system_instruction=AGENT_SYSTEM_INSTRUCTION,
    tools=[get_current_weather],   # the SDK builds the tool schema from the function
    temperature=0.0,
    # Crucial (Level 6): disable automatic calling so Gemini hands each call back to us.
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def _no_progress(message: str) -> None:
    """Default progress hook: do nothing."""


class WeatherCalls:
    """Runs the weather tool for ONE user request — shared by Gemini and Groq.

    - Only locations from the request are looked up.
    - Each location hits OpenWeather at most once. If Gemini already fetched a
      city before it was rate-limited, Groq gets the saved result instead of a
      second API call.
    - Calls run one at a time, in the order the LLM asks for them.
    """

    def __init__(self, locations: list[str], on_progress: Callable[[str], None] = _no_progress):
        self.expected_keys = {_city_key(loc) for loc in locations}
        self.fetched: dict[str, dict] = {}   # city key -> saved result
        self.readings: list[dict] = []       # real OpenWeather results, in call order
        self.on_progress = on_progress

    def run(self, tool_name: str, city: str) -> dict:
        city = (city or "").strip()
        key = _city_key(city)

        if tool_name != "get_current_weather":
            return {"ok": False, "error": f"Unknown tool '{tool_name}'."}
        if key not in self.expected_keys:
            return {"ok": False, "requested_location": city,
                    "error": "This location was not part of the request, so it was not looked up."}
        if key in self.fetched:
            self.on_progress(f"Reusing the weather already fetched for {city} (no new API call).")
            return self.fetched[key]

        self.on_progress(f"Getting weather for {city}…")   # shown right before the real call
        result = get_current_weather(city=city)
        self.fetched[key] = result
        self.readings.append(result)
        if result["ok"]:
            self.on_progress(f"✓ {result['city']}, {result['country']}: {result['temperature_c']:.2f} °C")
        else:
            self.on_progress(f"✗ {city}: {result['error']}")
        return result


def build_agent_prompt(user_prompt: str, locations: list[str]) -> str:
    ordered_list = "; ".join(f"{i}. {loc}" for i, loc in enumerate(locations, start=1))
    return (f"User request: {user_prompt}\n\n"
            f"Get the current weather for these locations, one at a time, in this order: {ordered_list}")


def run_gemini_agent(client: genai.Client, user_prompt: str, locations: list[str], weather: WeatherCalls) -> str:
    """Gemini manual function-calling loop (notebook Cell E).

    Returns Gemini's final summary (or a clear "[Agent stopped] ..." message).
    A rate-limit error is re-raised so run_with_failover() can switch to Groq.
    """
    # The conversation history we manage ourselves (Level 6).
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=build_agent_prompt(user_prompt, locations))])]

    for turn in range(1, MAX_AGENT_TURNS + 1):
        try:
            response = client.models.generate_content(model=MODEL_ID, contents=contents, config=AGENT_CONFIG)
        except Exception as exc:
            if is_rate_limit_error(exc):
                raise   # let run_with_failover() hand this request to Groq
            return f"[Agent stopped] {safe_gemini_error(exc)}"

        if not response.candidates or response.candidates[0].content is None:
            return "[Agent stopped] Gemini returned an empty response."

        # Keep Gemini's turn in the history exactly as returned (Level 6).
        contents.append(response.candidates[0].content)

        # No tool requested -> Gemini has given its final answer.
        function_calls = response.function_calls
        if not function_calls:
            return response.text or ""

        # Run each requested call, strictly one after another (plain for loop).
        response_parts = []
        for call in function_calls:
            city = str(dict(call.args or {}).get("city", ""))
            result = weather.run(call.name, city)
            response_parts.append(types.Part.from_function_response(name=call.name, response=result))

        # Send all results for this turn back together, as one message.
        contents.append(types.Content(role="user", parts=response_parts))

    return f"[Agent stopped] Reached the safety limit of {MAX_AGENT_TURNS} turns."


# ---------------------------------------------------------------------------
# Step 2 (fallback): the same agent loop on Groq
# ---------------------------------------------------------------------------
# Groq uses the OpenAI-style tool format, so the weather tool is described as JSON.
# It still runs the SAME get_current_weather() function through WeatherCalls.
GROQ_WEATHER_TOOL = {
    "type": "function",
    "function": {
        "name": "get_current_weather",
        "description": "Gets the current weather for one location from the OpenWeather API. "
                       "Call it once per location. Temperatures are in degrees Celsius.",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string",
                         "description": 'The location as "City,CountryCode", for example "Kathmandu,NP".'},
            },
            "required": ["city"],
        },
    },
}


def run_groq_agent(groq_client, groq_model: str, user_prompt: str, locations: list[str],
                   weather: WeatherCalls) -> str:
    """Groq version of the manual tool-calling loop. Never raises; errors become messages."""
    messages = [
        {"role": "system", "content": AGENT_SYSTEM_INSTRUCTION},
        {"role": "user", "content": build_agent_prompt(user_prompt, locations)},
    ]

    for _turn in range(1, MAX_AGENT_TURNS + 1):
        try:
            response = groq_client.chat.completions.create(
                model=groq_model,
                messages=messages,
                tools=[GROQ_WEATHER_TOOL],
                tool_choice="auto",
                parallel_tool_calls=False,   # ask Groq for one tool call per turn (sequential)
                temperature=0.0,
            )
        except Exception as exc:
            return f"[Agent stopped] {safe_groq_error(exc, groq_model)}"

        message = response.choices[0].message
        tool_calls = message.tool_calls or []
        if not tool_calls:
            return message.content or ""   # no tool requested -> final answer

        # Keep Groq's turn in the history, including the tool calls it asked for.
        messages.append({
            "role": "assistant",
            "content": message.content or "",
            "tool_calls": [{"id": tc.id, "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                           for tc in tool_calls],
        })

        # Run each requested call, strictly one after another (plain for loop).
        for tc in tool_calls:
            try:
                args = json.loads(tc.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            result = weather.run(tc.function.name, str(args.get("city", "")))
            messages.append({"role": "tool", "tool_call_id": tc.id, "content": json.dumps(result)})

    return f"[Agent stopped] Reached the safety limit of {MAX_AGENT_TURNS} turns."


# ---------------------------------------------------------------------------
# Step 3: summary & average (notebook Cell F — pure Python)
# ---------------------------------------------------------------------------
def summarize_readings(locations: list[str], readings: list[dict]) -> dict:
    """Matches each requested location (in order) to what the agent actually fetched."""
    results = []
    for location in locations:
        matches = [r for r in readings if _city_key(r.get("requested_location", "")) == _city_key(location)]
        successes = [r for r in matches if r.get("ok")]
        if successes:
            results.append({"location": location, **successes[-1]})
        elif matches:
            results.append({"location": location, **matches[-1]})
        else:
            results.append({"location": location, "ok": False,
                            "error": "The agent did not look up this location."})

    valid_temperatures = [r["temperature_c"] for r in results if r.get("ok")]
    return {
        "results": results,
        "valid_temperatures": valid_temperatures,
        "all_valid": len(valid_temperatures) == len(locations),
    }


def calculate_average_temperature(temperatures: list[float]) -> float:
    """Plain arithmetic mean: sum of the values divided by how many there are."""
    if not temperatures:
        raise ValueError("Cannot average an empty list of temperatures.")
    return sum(temperatures) / len(temperatures)


# ---------------------------------------------------------------------------
# One chat turn: prompt in -> reply dict out (stored in the chat history)
# ---------------------------------------------------------------------------
def run_with_failover(provider: dict, gemini_step: Callable, groq_step: Callable, groq_client, groq_model: str,
                      on_progress: Callable[[str], None]):
    """Runs one step on Gemini; on a Gemini rate-limit error, runs it on Groq instead.

    Gemini is our primary provider. If Gemini returns a rate-limit/quota error,
    we fall back to Groq so the chatbot can continue working. Once a request has
    switched to Groq, it stays on Groq (`provider["name"]` remembers this).
    Any other Gemini error is raised as-is — it is not a reason to switch.
    """
    if provider["name"] == "Gemini":
        try:
            return gemini_step()
        except Exception as exc:
            if not is_rate_limit_error(exc):
                raise
            on_progress("⚡ Gemini rate limit reached.")
            if groq_client is None:
                raise NoBackupError(
                    "Gemini's rate limit has been reached and no Groq backup is available "
                    "(check that GROQ_API_KEY is in .env and the groq package is installed). "
                    "Please wait a minute and try again.") from None
            on_progress("🔄 Switching to Groq…")
            provider["name"] = "Groq"
            provider["fell_back"] = True
            on_progress(f"🤖 Groq (`{groq_model}`) is handling this request.")
    return groq_step()


class NoBackupError(Exception):
    """Gemini is rate-limited and Groq can't be used. The message is safe to show."""


def handle_prompt(client: genai.Client, prompt: str, on_progress: Callable[[str], None] = _no_progress,
                  groq_client=None, groq_model: str = GROQ_MODEL_CANDIDATES[0]) -> dict:
    """Runs the whole pipeline for one prompt and returns a reply to display."""
    # Which LLM answered this request. Starts as Gemini; run_with_failover() may switch it.
    provider = {"name": "Gemini", "fell_back": False}

    def reply(**fields) -> dict:
        return {**fields, "provider": provider["name"], "fell_back": provider["fell_back"]}

    on_progress(f"🤖 Using Gemini (`{MODEL_ID}`)…")
    try:
        # Step 1: identify the locations (Gemini first, Groq on a Gemini rate limit).
        on_progress("Asking which locations you mentioned…")
        request, error = run_with_failover(
            provider,
            gemini_step=lambda: gemini_extract_locations(client, prompt),
            groq_step=lambda: groq_extract_locations(groq_client, groq_model, prompt),
            groq_client=groq_client, groq_model=groq_model, on_progress=on_progress)
        if error:
            return reply(kind="error", text=error)

        locations = unique_locations(request.locations)
        if not request.is_weather_request or not locations:
            return reply(kind="text", text=(
                "I'm a weather agent. Ask me for the current weather in up to "
                f"{MAX_LOCATIONS} places, for example: *Get the current weather for Kathmandu, London and Tokyo*."))
        if len(locations) > MAX_LOCATIONS:
            return reply(kind="text", text=(
                f"You asked about {len(locations)} locations ({', '.join(locations)}). "
                f"This agent currently supports up to {MAX_LOCATIONS} locations per request — "
                "please send a shorter list."))

        on_progress(f"Locations identified: {', '.join(locations)}")

        # Step 2: fetch the weather with tool calls (same failover rule).
        # If Gemini is rate-limited halfway through, Groq restarts the tool loop
        # (it can't continue Gemini's conversation), but `weather` remembers the
        # cities already fetched, so no OpenWeather call is made twice.
        weather = WeatherCalls(locations, on_progress)
        llm_text = run_with_failover(
            provider,
            gemini_step=lambda: run_gemini_agent(client, prompt, locations, weather),
            groq_step=lambda: run_groq_agent(groq_client, groq_model, prompt, locations, weather),
            groq_client=groq_client, groq_model=groq_model, on_progress=on_progress)
    except NoBackupError as exc:
        return reply(kind="error", text=str(exc))
    except Exception as exc:
        # A Gemini error that is NOT a rate limit (e.g. bad key): report it, don't fail over.
        text = safe_gemini_error(exc) if provider["name"] == "Gemini" else safe_groq_error(exc, groq_model)
        return reply(kind="error", text=text)

    # Step 3: Python calculates the authoritative numbers.
    summary = summarize_readings(locations, weather.readings)

    average = None
    if summary["all_valid"] and len(locations) >= 2:
        average = calculate_average_temperature(summary["valid_temperatures"])

    return reply(
        kind="weather",
        locations=locations,
        results=summary["results"],
        all_valid=summary["all_valid"],
        valid_temperatures=summary["valid_temperatures"],
        average=average,
        sequence=[r.get("requested_location", "") for r in weather.readings],
        gemini_text=llm_text,
    )


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def render_provider(reply: dict) -> None:
    """Small line showing which LLM produced this reply."""
    if reply.get("fell_back"):
        st.caption("🤖 Groq — used as fallback because Gemini's rate limit was reached")
    elif reply.get("provider"):
        st.caption(f"🤖 {reply['provider']}")


def render_reply(reply: dict) -> None:
    """Draws one assistant reply (used for new replies and for chat history)."""
    render_provider(reply)
    if reply["kind"] == "error":
        st.error(reply["text"])
        return
    if reply["kind"] == "text":
        st.markdown(reply["text"])
        return

    results = reply["results"]
    n = len(results)
    if reply["all_valid"]:
        st.markdown(f"Here's the current weather for {n} location{'s' if n != 1 else ''}:")
    else:
        st.markdown("I couldn't get a valid reading for every location:")

    rows = ["| # | Location | Temperature | Conditions |", "|---|---|---:|---|"]
    for i, r in enumerate(results, start=1):
        if r.get("ok"):
            rows.append(f"| {i} | {r['city']}, {r['country']} | {r['temperature_c']:.2f} °C "
                        f"| {r['description'].capitalize()} |")
        else:
            rows.append(f"| {i} | {r['location']} | — | ⚠️ {r['error']} |")
    st.markdown("\n".join(rows))

    temps = reply["valid_temperatures"]
    if reply["average"] is not None:
        working = " + ".join(f"{t:.2f}" for t in temps)
        st.metric(f"Average temperature ({n} locations)", f"{reply['average']:.2f} °C")
        st.caption(f"Calculated in Python: ({working}) / {n} = {reply['average']:.2f} °C")
    elif reply["all_valid"]:
        st.caption("Only one location was requested, so there is no average to calculate.")
    else:
        st.warning(f"Average not calculated: only {len(temps)} of {n} locations returned a valid "
                   "temperature. A partial average would be misleading.")

    if reply["sequence"]:
        order = " → ".join(f"{i}. {loc}" for i, loc in enumerate(reply["sequence"], start=1))
        st.caption(f"Tool calls, in order: {order}")

    gemini_text = (reply.get("gemini_text") or "").strip()
    if gemini_text.startswith("[Agent stopped]"):
        st.error(gemini_text.replace("[Agent stopped] ", "The agent stopped early: "))
    elif gemini_text:
        with st.expander(f"{reply.get('provider', 'Gemini')}'s summary"):
            st.markdown(gemini_text)
            st.caption("The temperatures and average above come from Python and OpenWeather.")


@st.cache_resource
def get_gemini_client() -> genai.Client:
    """Creates the Gemini client once and reuses it (the key is never displayed)."""
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


@st.cache_resource
def get_groq_client():
    """Creates the Groq backup client once, or returns None if Groq can't be used."""
    if groq is None or not os.getenv("GROQ_API_KEY"):
        return None
    return groq.Groq(api_key=os.getenv("GROQ_API_KEY"))


@st.cache_resource
def choose_groq_model(_groq_client) -> str:
    """Asks Groq (once) which models this key can use, and picks the first candidate available.

    Different Groq accounts/projects can have different models enabled, so a
    hard-coded model name can come back "not found". If the list can't be
    fetched, we just try the first candidate.
    """
    if _groq_client is None:
        return GROQ_MODEL_CANDIDATES[0]
    try:
        available = {model.id for model in _groq_client.models.list().data}
    except Exception:
        return GROQ_MODEL_CANDIDATES[0]
    for model_id in GROQ_MODEL_CANDIDATES:
        if model_id in available:
            return model_id
    return GROQ_MODEL_CANDIDATES[0]


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Weather Agent", page_icon="🌤️")
    st.title("🌤️ Weather Agent")
    st.caption(f"A Gemini-powered agent (`{MODEL_ID}`) that fetches current weather from OpenWeather, "
               f"one location at a time — up to {MAX_LOCATIONS} per request. "
               "Python calculates the average. If Gemini hits its rate limit, Groq takes over.")

    if SIMULATE_GEMINI_RATE_LIMIT:
        st.info("🧪 Test mode: every Gemini request raises a simulated rate-limit error, so Groq will answer. "
                "Restart without SIMULATE_GEMINI_RATE_LIMIT=1 to go back to normal.")

    # Check the API keys exist (names only — never the values).
    missing_keys = [name for name in REQUIRED_KEYS if not os.getenv(name)]
    if missing_keys:
        st.error(f"Missing API key(s) in `.env`: {', '.join(missing_keys)}. "
                 "Add them to the project's .env file and restart the app.")
        st.stop()

    with st.sidebar:
        st.markdown("**Try:**\n\n*Get the current weather for Kathmandu, London and Tokyo*")
        if st.button("Clear chat"):
            st.session_state.messages = []
        st.markdown("**Providers**")
        st.caption(f"Primary: Gemini (`{MODEL_ID}`)")
        if get_groq_client() is None:
            st.caption("Backup: Groq — not available (needs GROQ_API_KEY in .env and the groq package)")
        else:
            st.caption(f"Backup: Groq (`{choose_groq_model(get_groq_client())}`)")
        if st.session_state.get("last_provider"):
            st.caption(f"Last response: {st.session_state['last_provider']}")

    # Chat history lives in session_state for this browser session.
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages:
        with st.chat_message("assistant"):
            st.markdown(f"Hi! Ask me for the current weather in up to {MAX_LOCATIONS} places.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            if message["role"] == "user":
                st.markdown(message["content"])
            else:
                render_reply(message["content"])

    prompt = st.chat_input(f"Ask for the current weather in up to {MAX_LOCATIONS} places…")
    if not prompt:
        return

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.status("Working…", expanded=True) as status:
            reply = handle_prompt(get_gemini_client(), prompt, on_progress=st.write,
                                  groq_client=get_groq_client(),
                                  groq_model=choose_groq_model(get_groq_client()))
            ok = reply["kind"] != "error" and (reply["kind"] != "weather" or reply["all_valid"])
            label = "Done" if ok else "Finished with problems"
            if reply["fell_back"]:
                label += " (answered by Groq — Gemini was rate-limited)"
            status.update(label=label, state="complete" if ok else "error", expanded=False)
        render_reply(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})
    st.session_state["last_provider"] = reply["provider"] + (" (fallback)" if reply["fell_back"] else "")


# Streamlit runs this file as __main__; importing it (e.g. for tests) won't draw the page.
if __name__ == "__main__":
    main()
