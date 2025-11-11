# app.py
"""
Multi-Agent Travel Planner

Highlights:
- Clear separation of concerns (tools, agents, orchestration, UI)
- Simple global logger to display tool calls live in the sidebar
- Planner → Reviewer pipeline enforced before rendering any answer
- Minimal dependencies and straightforward control flow
"""

from __future__ import annotations

import os
import asyncio
import time
from typing import Callable, Dict, List, Optional, Any

import streamlit as st
from dotenv import load_dotenv
from tavily import TavilyClient

# ──────────────────────────────────────────────────────────────────────────────
# Environment & Globals
# ──────────────────────────────────────────────────────────────────────────────

load_dotenv()  # Loads variables from a local .env if present
os.environ.setdefault("OPENAI_LOG", "error")
os.environ.setdefault("OPENAI_TRACING", "false")

# Tool call logger: the UI sets this per request. The tool checks it and logs.
# Using a simple global makes this easy to teach and reason about.
TOOL_LOGGER: Optional[Callable[[Dict[str, Any]], None]] = None


def set_tool_logger(logger: Optional[Callable[[Dict[str, Any]], None]]) -> None:
    """Install or remove the UI logger used by tools to report activity."""
    global TOOL_LOGGER
    TOOL_LOGGER = logger


def log_tool_event(event: Dict[str, Any]) -> None:
    """If a logger is installed, send the event to the UI."""
    if TOOL_LOGGER is not None:
        try:
            TOOL_LOGGER(event)
        except Exception:
            # Logging should never break the app or the tool itself
            pass


def redact_for_logs(value: Any) -> Any:
    """
    Make sure we don't leak secrets and keep logs small.
    This is deliberately simple for teaching.
    """
    if isinstance(value, str):
        low = value.lower()
        if any(k in low for k in ("api_key", "token", "secret", "password")):
            return "[redacted]"
        return value if len(value) <= 300 else value[:120] + "… [truncated]"
    if isinstance(value, dict):
        return {k: ("[redacted]" if any(s in k.lower() for s in ("key", "token", "secret", "password"))
                    else redact_for_logs(v))
                for k, v in value.items()}
    if isinstance(value, list):
        return [redact_for_logs(v) for v in value]
    return value


# ──────────────────────────────────────────────────────────────────────────────
# Agent Framework Imports (provided by you)
# ──────────────────────────────────────────────────────────────────────────────
# These come from your own framework. We assume:
# - Agent: defines a model + instructions + optional tools
# - Runner.run(agent, input): executes an agent and returns an object with text
from agents import Agent, Runner, function_tool  # type: ignore


# ──────────────────────────────────────────────────────────────────────────────
# Tools
# ──────────────────────────────────────────────────────────────────────────────

@function_tool
def internet_search(query: str) -> str:
    """
    Internet search backed by Tavily.
    - Reads TAVILY_API_KEY from environment.
    - Sends simple log events before/after the call so the UI can show activity.
    """
    log_tool_event({"type": "call", "tool": "internet_search", "args": {"query": redact_for_logs(query)}})

    try:
        api_key = os.getenv("TAVILY_API_KEY")
        if not api_key:
            msg = "missing TAVILY_API_KEY in environment."
            log_tool_event({"type": "error", "tool": "internet_search", "error": msg})
            return f"Search error: {msg}"

        client = TavilyClient(api_key=api_key)
        response = client.search(query, max_results=3)

        items = response.get("results", [])
        lines = [f"- {it.get('title', 'N/A')}: {it.get('content', 'N/A')}" for it in items]
        output = "\n".join(lines) if lines else "No results found."

        log_tool_event({
            "type": "result",
            "tool": "internet_search",
            "preview": redact_for_logs(output[:400] + ("…" if len(output) > 400 else "")),
        })
        return output

    except Exception as e:
        log_tool_event({"type": "error", "tool": "internet_search", "error": str(e)})
        return f"Search error: {e}"

    finally:
        log_tool_event({"type": "end", "tool": "internet_search"})


# ──────────────────────────────────────────────────────────────────────────────
# Agents
# ──────────────────────────────────────────────────────────────────────────────

# BEGIN SOLUTION
REVIEWER_INSTRUCTIONS = """You are a meticulous travel itinerary reviewer and fact checker. Your role is to validate 
and improve travel plans created by the Planner agent by using real-time internet searches.

Core Mission:
By identifying problems and providing specific, actionable improvement suggestions (which must be based on verification), ensure that the itinerary is feasible, accurate and optimized

Your Responsibilities:

1. The verification information must be verified through inrernet search:
   You must use the internet_search tool to verify:
   - The business hours and opening dates of the scenic spots
   - Current ticket prices and booking requirements
   - Seasonal opening situation
   - The transportation between locations takes a long time
   - Special closure or event information
   - Restaurant business hours
   - Mode of transportation and costs

   Search Strategy:
   - Use precise queries, such as "Louvre Operating Hours in 2024"
   - Search for each major attraction separately
   - Verify at least 3 to 5 key pieces of information every day
   - Check the transportation and logistics between locations

2. Identify Feasibility Issues:
   - Time issue: The schedule is too tight or the transportation time is not realistic
   - Closed Venues: The attraction will be closed on the planned date
   - Budget Errors: Incorrect prices or missing costs
   - Route Issues: Retracing one's steps or taking an inefficient path
   - Seasonal Problems: Activities unavailable in that season
   - Missing Information: There are no reservation instructions for popular attractions
   - Time conflict: It is impossible for activities to overlap or connect

3. Create a Structured Review:

Use this format:
---
REVIEW SUMMARY
Overall Assessment: [Excellent/Good/Needs Revision]
Total Issues Found: [Number]
Facts Verified: [Number of searches conducted]
---
DELTA LIST - Required Changes
Issue #[X]: [Brief title]
- Problem: [What's wrong in the itinerary]
- Evidence: [What you discovered via internet search]
- Impact: [How this affects the trip]
- Suggested Fix: [Specific replacement or adjustment]
- Reasoning: [Why this fix is better]
---
POSITIVE ASPECTS
- [List 2-3 things the Planner did well]
---
VERIFIED FACTS
- [List key facts you verified through internet search]
---
FINAL RECOMMENDATIONS
[Provide 2-3 summary recommendations]
---

4. Search Best Practices:
   - Be thorough but efficient
   - Prioritize verification of major attractions, fees and business hours
   - Search for information that may have changed recently
   - Record your search process so that users can see the verification trajectory

5. Tone Guidelines:
   - Constructive feedback, don't just criticize
   - Explain why it will be better after the modification
   - Acknowledge good planning decisions
   - Offer feasible solutions, not just point out the problem

6. Final output format:
   - Start with a brief validation summary (1-2 sentences on overall feasibility)
   - List all issues with their fixes in the Delta List
   - If no major issues, say "Itinerary is validated with minor suggestions" or similar
   - Provide the REVISED ITINERARY with all fixes integrated
   - Maintain the original structure and detail level

7. Quality checks:
   - Ensure total cost still aligns with budget after fixes
   - Verify all logistics are realistic and achievable
   - Confirm activities are actually feasible on their scheduled days
   - Check for cultural sensitivity and practical considerations

Critical Instruction: 
You MUST use the internet_search tool multiple times. A good review includes 5-20 searches 
depending on itinerary complexity. Do not rely on your training data - verify current information.
"""



PLANNER_INSTRUCTIONS = """You are an expert travel planner with extensive knowledge of destinations worldwide. 
Your role is to create detailed, realistic, and exciting travel itineraries based on user requests.


1. Create a detailed daily schedule:
   - Break down each day into Morning, Afternoon, and Evening 
   - Include specific activities with approximate times
   - Provide specific locations with addresses or landmarks where relevant
   - Estimate costs for each activity (entrance fees, meals, transportation)
   - Include transportation details between locations with estimated travel times
   - Group activities into logical city clusters to minimize unnecessary travel

2. Fully consider user requirements:
   - Budget: Track spending day-by-day and provide running totals. Stay within the stated budget.
   - Duration: Respect the trip length (e.g., 3 days, one week)
   - Interests: Tailor activities to user preferences (history, food, adventure, nature, culture, etc.)
   - Pacing: Balance activities - don't overpack. Include time for rest, meals, and spontaneous exploration
   - Travel Style: Consider if user is budget traveler, luxury seeker, family-friendly, solo adventurer, etc.

3. Provide Practical Information:
   - Booking recommendations (e.g., "Book Eiffel Tower tickets 2-3 weeks in advance")
   - Best times to visit attractions (e.g., "Visit early morning to avoid crowds")
   - Local transportation options (metro, bus, walking distances)
   - Meal suggestions with price ranges
   - Safety tips or cultural etiquette notes when relevant

4. Format Your Output Clearly:

Use this structure for each day:
---
DAY [X] - [City/Location Name]
Morning (9:00 AM - 12:00 PM):
• 9:00 AM - Activity Name
  - Location: [Specific address or landmark]
  - Cost: $[Amount]
  - Duration: [Time estimate]
  - Description: [Brief description and why it's worth visiting]

Afternoon (12:00 PM - 6:00 PM):
• 12:00 PM - Lunch at [Restaurant/Area]**
  - Location: [Place]
  - Cost: $[Amount]
  - Description: [Cuisine type and recommendations]

• 2:00 PM - Activity Name
  - Location: [Specific address or landmark]
  - Cost: $[Amount]
  - Duration: [Time estimate]
  - Description: [Brief description]

Evening (6:00 PM - 10:00 PM):
• 6:00 PM - Activity Name
  - Location: [Place]
  - Cost: $[Amount]
  - Description: [Brief description]

Daily Budget:
  - Activities: $[X]
  - Food: $[X]
  - Transportation: $[X]
  - Daily Total: $[X]
  
Running Trip Total: $[X]
---
5. End with a Trip Summary:
   - Total estimated cost with breakdown
   - Essential booking information
   - Overall tips and recommendations

Notices:
- You have no network access - Work based on the existing knowledge base
- There will be review experts to verify your plan later
- Time arrangement should be realistic - consider travel time, queuing and rest
- Balance famous attractions with hidden gems
- Make the itinerary attractive and personalized based on users' interests
"""

# Create the agents with proper tool assignment
reviewer_agent = Agent(
    name="Reviewer Agent",
    model="openai.gpt-4o",
    instructions=REVIEWER_INSTRUCTIONS.strip(),
    tools=[internet_search]  # ← CRITICAL: Reviewer needs the search tool
)

planner_agent = Agent(
    name="Planner Agent",
    model="openai.gpt-4o",
    instructions=PLANNER_INSTRUCTIONS.strip(),
    # No tools - Planner works from knowledge only
)

# END SOLUTION


# ──────────────────────────────────────────────────────────────────────────────
# Orchestration Helpers
# ──────────────────────────────────────────────────────────────────────────────

def extract_text(result_obj: Any) -> str:
    """
    Pull a usable string from the Runner result in a tolerant way.
    Your Runner may expose final_output, text, or __str__.
    """
    return (
        getattr(result_obj, "final_output", None)
        or getattr(result_obj, "text", None)
        or str(result_obj)
    )


def run_planner(user_text: str) -> str:
    """Run the Planner and return its itinerary text."""
    result = asyncio.run(Runner.run(planner_agent, user_text))
    return extract_text(result)


def run_reviewer(plan_text: str) -> str:
    """Run the Reviewer on the planner’s output and return validated text."""
    result = asyncio.run(Runner.run(reviewer_agent, plan_text))
    return extract_text(result)


# ──────────────────────────────────────────────────────────────────────────────
# Streamlit UI
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(page_title="Travel Planner", page_icon="✈️")

st.title("✈️ Multi-Agent Travel Planner")
st.caption("Planner → Reviewer (with live tool calls in the sidebar)")

# Sidebar: session controls + examples + dev panel
with st.sidebar:
    st.header("Session")
    if st.button("🔄 Reset conversation"):
        st.session_state.clear()
        st.rerun()

    st.subheader("Try these prompts")
    st.code("Plan a week-long Europe trip for a student on a $1,500 budget who loves history and food")
    st.code("3-day Paris trip for art lovers with $800 budget")

    st.subheader("Developer view")
    show_tools = st.toggle("Show tool activity (live)", value=True)
    if show_tools:
        tool_expander = st.expander("🔧 Tool activity", expanded=True)
        tool_panel = tool_expander.container()
    else:
        tool_panel = st.container()  # inert sink

# Session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []  # list[dict(role, content)]
if "meta" not in st.session_state:
    st.session_state.meta = []      # list[dict(trace)]

# Render history
for i, msg in enumerate(st.session_state.messages):
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg["role"] == "assistant" and i < len(st.session_state.meta):
            meta = st.session_state.meta[i]
            if meta:
                st.caption(meta.get("trace", ""))

# Chat input
user_input = st.chat_input("Describe your travel (destination, duration, budget, interests)…")

if user_input:
    # Add user message to history and render it
    st.session_state.messages.append({"role": "user", "content": user_input})
    st.session_state.meta.append(None)
    with st.chat_message("user"):
        st.markdown(user_input)

    # Assistant output block
    with st.chat_message("assistant"):
        # Live “working…” text and progress bar
        live_msg = st.empty()
        progress = st.progress(0)

        # Per-request tool log (shown in the sidebar)
        tool_events: List[Dict[str, Any]] = []

        def ui_tool_logger(event: Dict[str, Any]) -> None:
            """Append an event and re-render the sidebar log."""
            tool_events.append(event)
            with tool_panel:
                st.markdown("**Recent tool calls**")
                for ev in tool_events[-60:]:  # last N entries
                    t = ev.get("tool", "unknown")
                    et = ev.get("type", "event")
                    if et == "call":
                        st.write(f"• **{t}** called with `{ev.get('args')}`")
                    elif et == "result":
                        st.write(f"• **{t}** result preview:\n\n> {ev.get('preview')}")
                    elif et == "error":
                        st.error(f"• **{t}** error: {ev.get('error')}")
                    elif et == "end":
                        st.write(f"• **{t}** finished")

        # Install the logger so tools can report to the sidebar
        set_tool_logger(ui_tool_logger)

        try:
            # Optional: clear sidebar panel on each run
            with tool_panel:
                st.empty()

            # Step 1: Planner
            with st.status("🧭 Planner Agent: generating itinerary…", expanded=True) as status:
                live_msg.markdown("🧭 Planner Agent is creating your itinerary…")
                plan_text = run_planner(user_input)
                progress.progress(40)
                status.update(label="🔎 Reviewer Agent: validating with live searches…", state="running")

            # Step 2: Reviewer (tool calls will appear live in sidebar)
            live_msg.markdown("🔎 Reviewer Agent is validating the plan with live searches…")
            review_text = run_reviewer(plan_text)
            progress.progress(90)

            # Completed
            live_msg.markdown("Validation complete. Rendering results…")
            time.sleep(0.2)
            progress.progress(100)

            # Final render: show only the validated result, with the raw plan expandable
            st.info("Reviewer Agent (validated)")
            st.markdown(review_text)
            with st.expander("See raw plan from Planner Agent"):
                st.markdown(plan_text)

            # Save only the validated result to history
            st.session_state.messages.append({"role": "assistant", "content": review_text})
            st.session_state.meta.append({"trace": "Planner Agent → Reviewer Agent"})
            st.caption("Planner Agent → Reviewer Agent")

        except Exception as e:
            # Friendly error box
            live_msg.markdown("❌ Something went wrong.")
            err = f"⚠️ Error while processing your request:\n\n```\n{e}\n```"
            st.markdown(err)
            st.session_state.messages.append({"role": "assistant", "content": err})
            st.session_state.meta.append({"trace": "Runtime error."})

        finally:
            # Always remove the logger so it doesn't leak into the next request
            set_tool_logger(None)
