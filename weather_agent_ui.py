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

Start it from the project folder with:
    .venv/bin/python -m streamlit run weather_agent_ui.py

API keys are read from the project's .env file and are never displayed.
"""

import os
from pathlib import Path
from typing import Callable, Optional

import requests
import streamlit as st
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
from pydantic import BaseModel, Field, ValidationError

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
# Load .env from the folder this file lives in, so it works from any directory.
load_dotenv(Path(__file__).resolve().parent / ".env")

MODEL_ID = "gemini-3.6-flash"
MAX_LOCATIONS = 3              # Assignment 1 is based on three locations
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"
UNITS = "metric"               # "metric" -> temperatures in degrees Celsius
REQUEST_TIMEOUT_SECONDS = 10   # give up on a slow OpenWeather request
MAX_AGENT_TURNS = 6            # safety limit so the agent loop can't run forever
REQUIRED_KEYS = ["GEMINI_API_KEY", "OPENWEATHER_API_KEY"]


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
    if code in (401, 403):
        return "Gemini rejected the API key or permissions."
    if code == 404:
        return f"Gemini model '{MODEL_ID}' was not found."
    if code == 429:
        return "Gemini rate limit reached. Please wait a moment and try again."
    if isinstance(code, int) and code >= 500:
        return "Gemini is temporarily unavailable. Please try again."
    if isinstance(exc, errors.APIError):
        return f"Gemini could not process the request (error {code})."
    return f"Could not reach Gemini ({type(exc).__name__})."


# ---------------------------------------------------------------------------
# Step 1: Gemini identifies the locations (structured output, notebook Level 3)
# ---------------------------------------------------------------------------
class LocationRequest(BaseModel):
    is_weather_request: bool = Field(
        description="True if the user is asking for the current weather of one or more places.")
    locations: list[str] = Field(
        description='Every place the user asked about, in the order mentioned, formatted as '
                    '"City,CountryCode" (e.g. "Kathmandu,NP"). Use just "City" if the country is unclear.')


EXTRACTION_CONFIG = types.GenerateContentConfig(
    system_instruction=(
        "You extract locations from a user's weather request. List every place the user "
        "asked about, in the order they were mentioned, with no duplicates. Add the ISO "
        "3166 two-letter country code only when you are confident of it. Do not add places "
        "the user did not mention, and do not answer the question yourself."
    ),
    response_mime_type="application/json",
    response_schema=LocationRequest,
    temperature=0.0,
    automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
)


def extract_locations(client: genai.Client, prompt: str) -> tuple[Optional[LocationRequest], Optional[str]]:
    """Asks Gemini which locations the prompt mentions. Returns (request, error_message)."""
    try:
        response = client.models.generate_content(model=MODEL_ID, contents=prompt, config=EXTRACTION_CONFIG)
    except Exception as exc:
        return None, safe_gemini_error(exc)

    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, LocationRequest):
        return parsed, None
    try:
        return LocationRequest.model_validate_json(response.text or ""), None
    except (ValidationError, ValueError):
        return None, "Gemini did not return a readable list of locations. Please rephrase your request."


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


def run_weather_agent(
    client: genai.Client,
    user_prompt: str,
    locations: list[str],
    on_progress: Callable[[str], None] = _no_progress,
) -> tuple[list[dict], str]:
    """Runs the manual function-calling loop for the given locations.

    Python only runs a weather request for a location that is in `locations`
    and hasn't been fetched yet, so there are at most len(locations) real
    OpenWeather calls, always one after another.

    Returns:
        (readings, gemini_text): every real tool result in the order it happened,
        and Gemini's final summary (or a clear "[Agent stopped] ..." message).
    """
    readings: list[dict] = []
    expected_keys = {_city_key(loc) for loc in locations}
    fetched_keys: set[str] = set()

    ordered_list = "; ".join(f"{i}. {loc}" for i, loc in enumerate(locations, start=1))
    prompt = (f"User request: {user_prompt}\n\n"
              f"Get the current weather for these locations, one at a time, in this order: {ordered_list}")

    # The conversation history we manage ourselves (Level 6).
    contents = [types.Content(role="user", parts=[types.Part.from_text(text=prompt)])]

    for turn in range(1, MAX_AGENT_TURNS + 1):
        try:
            response = client.models.generate_content(model=MODEL_ID, contents=contents, config=AGENT_CONFIG)
        except Exception as exc:
            return readings, f"[Agent stopped] {safe_gemini_error(exc)}"

        if not response.candidates or response.candidates[0].content is None:
            return readings, "[Agent stopped] Gemini returned an empty response."

        # Keep Gemini's turn in the history exactly as returned (Level 6).
        contents.append(response.candidates[0].content)

        # No tool requested -> Gemini has given its final answer.
        function_calls = response.function_calls
        if not function_calls:
            return readings, (response.text or "")

        # Run each requested call, strictly one after another (plain for loop).
        response_parts = []
        for call in function_calls:
            city = str(dict(call.args or {}).get("city", "")).strip()
            key = _city_key(city)

            if call.name != "get_current_weather":
                result = {"ok": False, "error": f"Unknown tool '{call.name}'."}
            elif key not in expected_keys:
                result = {"ok": False, "requested_location": city,
                          "error": "This location was not part of the request, so it was not looked up."}
            elif key in fetched_keys:
                result = {"ok": False, "requested_location": city,
                          "error": "This location was already fetched."}
            else:
                on_progress(f"Getting weather for {city}…")   # shown right before the real call
                result = get_current_weather(city=city)
                readings.append(result)
                fetched_keys.add(key)
                if result["ok"]:
                    on_progress(f"✓ {result['city']}, {result['country']}: {result['temperature_c']:.2f} °C")
                else:
                    on_progress(f"✗ {city}: {result['error']}")

            response_parts.append(types.Part.from_function_response(name=call.name, response=result))

        # Send all results for this turn back together, as one message.
        contents.append(types.Content(role="user", parts=response_parts))

    return readings, f"[Agent stopped] Reached the safety limit of {MAX_AGENT_TURNS} turns."


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
def handle_prompt(client: genai.Client, prompt: str, on_progress: Callable[[str], None] = _no_progress) -> dict:
    """Runs the whole pipeline for one prompt and returns a reply to display."""
    on_progress("Asking Gemini which locations you mentioned…")
    request, error = extract_locations(client, prompt)
    if error:
        return {"kind": "error", "text": error}

    locations = unique_locations(request.locations)
    if not request.is_weather_request or not locations:
        return {"kind": "text", "text": (
            "I'm a weather agent. Ask me for the current weather in up to "
            f"{MAX_LOCATIONS} places, for example: *Get the current weather for Kathmandu, London and Tokyo*.")}
    if len(locations) > MAX_LOCATIONS:
        return {"kind": "text", "text": (
            f"You asked about {len(locations)} locations ({', '.join(locations)}). "
            f"This agent currently supports up to {MAX_LOCATIONS} locations per request — "
            "please send a shorter list.")}

    on_progress(f"Locations identified: {', '.join(locations)}")
    readings, gemini_text = run_weather_agent(client, prompt, locations, on_progress)
    summary = summarize_readings(locations, readings)

    average = None
    if summary["all_valid"] and len(locations) >= 2:
        average = calculate_average_temperature(summary["valid_temperatures"])

    return {
        "kind": "weather",
        "locations": locations,
        "results": summary["results"],
        "all_valid": summary["all_valid"],
        "valid_temperatures": summary["valid_temperatures"],
        "average": average,
        "sequence": [r.get("requested_location", "") for r in readings],
        "gemini_text": gemini_text,
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------
def render_reply(reply: dict) -> None:
    """Draws one assistant reply (used for new replies and for chat history)."""
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
        with st.expander("Gemini's summary"):
            st.markdown(gemini_text)
            st.caption("The temperatures and average above come from Python and OpenWeather.")


@st.cache_resource
def get_gemini_client() -> genai.Client:
    """Creates the Gemini client once and reuses it (the key is never displayed)."""
    return genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


# ---------------------------------------------------------------------------
# The page
# ---------------------------------------------------------------------------
def main() -> None:
    st.set_page_config(page_title="Weather Agent", page_icon="🌤️")
    st.title("🌤️ Weather Agent")
    st.caption(f"A Gemini-powered agent (`{MODEL_ID}`) that fetches current weather from OpenWeather, "
               f"one location at a time — up to {MAX_LOCATIONS} per request. "
               "Python calculates the average.")

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
            reply = handle_prompt(get_gemini_client(), prompt, on_progress=st.write)
            ok = reply["kind"] != "error" and (reply["kind"] != "weather" or reply["all_valid"])
            status.update(label="Done" if ok else "Finished with problems",
                          state="complete" if ok else "error", expanded=False)
        render_reply(reply)

    st.session_state.messages.append({"role": "assistant", "content": reply})


# Streamlit runs this file as __main__; importing it (e.g. for tests) won't draw the page.
if __name__ == "__main__":
    main()
