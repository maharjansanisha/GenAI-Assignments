# GenAI Assignments

Assignments for the GenAI course, built with Python, the **Google Gemini SDK** and **Strands Agents**.

| #   | Assignment                                                                     | Files                                                     | Status |
| --- | ------------------------------------------------------------------------------ | --------------------------------------------------------- | ------ |
| 1   | [Weather Agent](#assignment-1-weather-agent)                                   | `Assignment01_Weather_Agent.ipynb`, `weather_agent_ui.py` | Done   |
| 2   | [Strands Travel Planner](#assignment-2-strands-travel-planner)                 | `Assignment02_Strands_Travel_Planner.ipynb`               | Done   |
| 3   | [Strands Travel Planner Chatbot](#assignment-3-strands-travel-planner-chatbot) | `assignment03_travel_chatbot.py`                          | Done   |

## Setup (shared by all assignments)

Requires Python 3.12.

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Create a `.env` file in the project folder

```text
GEMINI_API_KEY=your-gemini-key
OPENWEATHER_API_KEY=your-openweather-key
GROQ_API_KEY=your-groq-key       # Assignments 2 and 3 (and the optional failover in Assignment 1)
TAVILY_API_KEY=your-tavily-key   # Assignments 2 and 3 web search
```

Keys: [Google AI Studio](https://aistudio.google.com/apikey) · [OpenWeather](https://home.openweathermap.org/api_keys) · [Groq](https://console.groq.com/keys) · [Tavily](https://app.tavily.com)

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

---

## Assignment 2: Strands Travel Planner

A **Strands Agents** agent that creates a one-day travel plan for a city. It uses three tools: current weather, web search for 3 popular attractions and their entry fees, and a calculator for the total cost.

| File                                        | What it is                                    |
| ------------------------------------------- | --------------------------------------------- |
| `Assignment02_Strands_Travel_Planner.ipynb` | The solution notebook (tested with Kathmandu) |

### How it works

1. **Model:** Groq's `openai/gpt-oss-120b`, used through Strands' `OpenAIModel` with Groq's OpenAI-compatible endpoint.
2. **Tools** are plain Python functions with the `@tool` decorator:
   - `get_current_weather(city)` — OpenWeather, in °C
   - `search_web(query)` — Tavily web search (the agent writes its own queries for attractions and entry fees)
   - `calculate_total_cost(attractions, costs, currency)` — adds the entry fees of exactly 3 attractions in Python, showing each attraction next to its cost
3. **Sequential tool calls:** `SequentialToolExecutor()` runs one tool at a time, so the calculator only runs after the prices have been found.
4. **The total comes from the calculator**, not the model. The notebook prints the tool calls the agent made, the calculator's result and the final itinerary separately, so they can be checked against each other.

### Run

Open the notebook, select the `.venv` kernel and run all cells. Change `CITY` in Section 6 to plan a different city.

### Error handling

- An unknown city, a failed search or invalid costs return a clear error from the tool instead of crashing.
- The agent run is wrapped in `try/except`, so a rate limit or network problem shows a friendly message.
- If an attraction's price can't be found, the plan says so and it counts as 0 in the total.

---

## Assignment 3: Strands Travel Planner Chatbot

A **Gradio chatbot** version of Assignment 2: type a city and get a one-day travel plan in the chat.

| File                             | What it is                                |
| -------------------------------- | ----------------------------------------- |
| `assignment03_travel_chatbot.py` | The chatbot app (Strands + Groq + Gradio) |

### How it works

1. **Same agent as Assignment 2:** Groq's `openai/gpt-oss-120b` through Strands' `OpenAIModel`, with the weather, Tavily search and calculator tools, run one at a time with `SequentialToolExecutor()`.
2. **Gradio UI:** `gr.ChatInterface` calls a `chat(message, history)` function, as in the instructor's `GenAI03_chatbot.py`.
3. **A fresh agent for each message**, so one city's plan doesn't affect the next.
4. **Each reply ends with a footer** showing the tools used and the calculator's result, e.g. `Pashupatinath Temple (1000) + Boudhanath Stupa (400) + Swayambhunath (200) = 1600 NPR`.

### Run

```bash
.venv/bin/python assignment03_travel_chatbot.py
```

Then open http://127.0.0.1:7860 and type a city, e.g. _Kathmandu_ or _Plan a day in Pokhara_. Each plan takes about 20–40 seconds. Stop the app with Ctrl+C.

### Error handling

- An empty message or a message without a city gets a friendly prompt to name one.
- An unknown city gets a "check the spelling" reply.
- A Groq, network or search failure shows a "please try again" message instead of crashing.
