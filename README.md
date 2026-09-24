# GenAI Assignments

Assignments for the GenAI course, built with the **Google Gemini SDK** and Python.

| #   | Assignment                                    | Files                                                     | Status  |
| --- | --------------------------------------------- | --------------------------------------------------------- | ------- |
| 1   | [Weather Agent](#assignment-1--weather-agent) | `Assignment01_Weather_Agent.ipynb`, `weather_agent_ui.py` | ✅ Done |
| 2   | _Strands Travel Planner_                      |                                                           |         |
| 3   | _Strands Travel Planner Chatbot_              |                                                           |         |

## Setup (shared by all assignments)

Requires Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Create a `.env` file in the project folder (it is git-ignored — never commit it):

```text
GEMINI_API_KEY=your-gemini-key
OPENWEATHER_API_KEY=your-openweather-key
GROQ_API_KEY=your-groq-key   # optional
```

Keys: [Google AI Studio](https://aistudio.google.com/apikey) · [OpenWeather](https://home.openweathermap.org/api_keys) · [Groq](https://console.groq.com/keys)

To run a notebook, open it, select the `.venv` kernel, and run the cells top to bottom.

---

## Assignment 1: Weather Agent

A Gemini agent that gets the current weather for up to three locations from the **OpenWeather API**, calls the weather tool **one location at a time**, and calculates the **average temperature in Python**.

| File                               | What it is                                                           |
| ---------------------------------- | -------------------------------------------------------------------- |
| `Assignment01_Weather_Agent.ipynb` | The solution notebook (Kathmandu, London, Tokyo)                     |
| `weather_agent_ui.py`              | A Streamlit chat app for the same agent, with Gemini → Groq failover |

### How it works

1. **Gemini reads the request** and identifies the locations.
2. **Manual function calling:** Gemini asks for `get_current_weather(city)`, and Python runs the real OpenWeather request and sends the result back. Automatic function calling is turned off, so every call is visible and controlled.
3. **Sequential calls:** one location per turn — for example Kathmandu → London → Tokyo — never in parallel.
4. **Python calculates the average** from the actual readings. It is only shown when every location returns a valid temperature, so a failed lookup never produces a misleading partial average.
5. **Failover (chat app only):** if Gemini returns a rate-limit/quota error (HTTP 429 / `RESOURCE_EXHAUSTED`), the request is handed to **Groq**, which uses the same weather tool. Other errors (for example an invalid key) are reported, not failed over.

Models: Gemini `gemini-3.6-flash` (primary). For Groq, the app uses the first of `llama-3.3-70b-versatile`, `openai/gpt-oss-120b`, `openai/gpt-oss-20b`, `llama-3.1-8b-instant` that your key can access.

### Run the chat app

```bash
.venv/bin/python -m streamlit run weather_agent_ui.py
```

Then open http://localhost:8501 and type, for example: _Get the current weather for Kathmandu, London and Tokyo_.

**Test the Groq failover** without using up your Gemini quota:

```bash
SIMULATE_GEMINI_RATE_LIMIT=1 .venv/bin/python -m streamlit run weather_agent_ui.py
```

A "Test mode" banner appears and every request goes to Groq. Restart without the variable to return to normal.

### Error handling

- Unknown cities, timeouts, bad keys and rate limits return clear messages instead of crashing.
- API keys are read from `.env` only and are never printed or shown. OpenWeather error text is never passed on, because its request URL contains the key.
- The chat app accepts up to 3 locations per request.
