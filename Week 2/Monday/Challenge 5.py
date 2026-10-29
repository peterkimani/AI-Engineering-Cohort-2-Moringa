# challenge5_stateful_patient_assistant.py
# Challenge 5: The Stateful, Multi-Tool Patient Assistant
#
# Requires: langchain-classic, langchain-openai, langchain-community, python-dotenv

import os                                      # standard library, used for environment access
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_openai import ChatOpenAI        # the OpenAI chat model wrapper
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor  # agent builder + run loop
from langchain_core.prompts import ChatPromptTemplate  # builds the agent's prompt with placeholders
from langchain_core.tools import tool          # the @tool decorator that turns a function into a tool
from langchain_core.runnables.history import RunnableWithMessageHistory  # adds per-session memory
from langchain_community.chat_message_histories import ChatMessageHistory  # stores one session's messages

# Load the .env file (for example OPENAI_API_KEY) into the environment so the client finds the key.
load_dotenv()


# ---------------------------------------------------------------------------
# Step 1: Multiple tools
# ---------------------------------------------------------------------------

# TOOL 1: stock lookup. The docstring says ONLY use it for stock and NOT for
# referrals or locations, so the agent does not confuse it with the other tools.
@tool
def get_clinic_stock_count(medication_name: str) -> str:
    """Use this tool ONLY to check how many units of a medication are in stock
    at the AfyaPlus clinic pharmacy. Input is a medication name such as
    'amoxicillin' or 'paracetamol'. Do not use it for referrals or
    department locations.

    How it works: the name is lowercased and stripped of surrounding spaces,
    then looked up in a hardcoded stock dictionary. If the medication is not
    listed, the count defaults to 0. The result is returned as a string such
    as "120 units".
    """
    # Mock inventory. A real system would query a database here instead.
    stock = {"amoxicillin": 120, "paracetamol": 540, "ibuprofen": 75}
    # Normalise the input so "Amoxicillin " matches the key; .get(..., 0) gives 0 if absent.
    count = stock.get(medication_name.strip().lower(), 0)
    return f"{count} units"


# TOOL 2: specialist referral. Its docstring contrasts with the other two so
# the agent sends symptom descriptions here, not stock or location questions.
@tool
def refer_to_specialist(symptoms: str) -> str:
    """Use this tool ONLY when a patient describes symptoms or a condition and
    you need to decide WHICH medical specialist they should be referred to.
    Input is a short free-text description of the symptoms. Do not use it for
    medication stock or for finding where a department is located.

    How it works: the symptom text is lowercased and scanned for keywords.
    Each specialty has a keyword list, and the specialty with the most
    keyword matches is chosen. If no keyword matches, the tool advises a
    general practitioner review instead of a specialist referral.
    """
    # Each specialty maps to the symptom keywords that point to it.
    keyword_map = {
        "cardiology": ["chest pain", "palpitation", "heart", "high blood pressure", "shortness of breath"],
        "dermatology": ["rash", "skin", "itch", "eczema", "acne"],
        "pediatrics": ["child", "infant", "baby", "toddler", "my son", "my daughter"],
    }
    text = symptoms.lower()  # lowercase once so keyword matching is case-insensitive

    # Score each specialty by counting how many of its keywords appear in the text
    scores = {
        specialty: sum(1 for kw in keywords if kw in text)
        for specialty, keywords in keyword_map.items()
    }
    # max with key=scores.get compares the counts and returns the top-scoring specialty.
    best_specialty = max(scores, key=scores.get)

    # A top score of 0 means nothing matched, so fall back to a GP review.
    if scores[best_specialty] == 0:
        return "No clear specialist match. Recommend a general practitioner review first."
    return f"Refer the patient to {best_specialty}."


# TOOL 3: department lookup. Used for where a department is, not which one is needed.
@tool
def lookup_specialist_department(specialty_name: str) -> str:
    """Use this tool ONLY to find the location, opening hours or booking status
    of a named medical department at AfyaPlus (for example 'cardiology').
    Input is a specialty name. Do not use it to decide which specialist a
    patient needs.

    How it works: the specialty name is lowercased and stripped of spaces,
    then looked up in a roster dictionary. If it is not found, a message
    saying the department is not listed is returned.
    """
    # Mock roster of department locations and status.
    roster = {
        "pediatrics": "Located in Wing A, open until 8 PM.",
        "cardiology": "Located in Main Tower, requires pre-booking.",
        "dermatology": "Nairobi Hub clinic, fully booked this week.",
    }
    # Return the matching entry, or a not-listed message if the key is absent.
    return roster.get(specialty_name.strip().lower(), "Department not listed at AfyaPlus.")


# Create the reasoning model. temperature=0.0 keeps tool choice consistent.
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.0)
# All three tools are registered on one agent, so it picks the right one per question.
tools = [get_clinic_stock_count, refer_to_specialist, lookup_specialist_department]


# ---------------------------------------------------------------------------
# Step 2: Prompt with a chat_history placeholder (short-term memory slot)
# ---------------------------------------------------------------------------

# Two placeholders matter here:
#   {chat_history}    is where past turns are replayed, giving the agent memory.
#   {agent_scratchpad} is where the reason-act loop records its tool calls.
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AfyaPlus patient assistant. Remember the patient's name and "
     "details across turns, and use tools whenever stock, referral or "
     "department information is needed."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])


# ---------------------------------------------------------------------------
# Step 3: Construct the chain and wrap with RunnableWithMessageHistory
# ---------------------------------------------------------------------------

# In-memory store of chat histories, keyed by session ID. Each session ID gets
# its own ChatMessageHistory, so different patients do not share memory.
store = {}


def get_session_history(session_id: str):
    """Returns the chat history object for a given session ID.

    Uses the module-level `store` dict as a lookup keyed by session_id. If no
    history exists for that ID yet, a new empty ChatMessageHistory is created
    and saved. The same object is returned on later calls, so messages
    accumulate across turns. Nothing is written to disk, so history is lost
    when the script exits.
    """
    # First time we see this session ID, create an empty history for it.
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    # Return the (new or existing) history so messages keep accumulating.
    return store[session_id]


# Build the agent and wrap it in an executor (called "chain" here).
agent = create_tool_calling_agent(llm, tools, prompt)
chain = AgentExecutor(agent=agent, tools=tools, verbose=True)

# RunnableWithMessageHistory adds memory around the executor:
#   input_messages_key   tells it which input field is the new user message.
#   history_messages_key tells it which prompt placeholder to fill with past turns.
# Before each call it loads the session's history into {chat_history};
# after each call it saves the new human and AI messages back to the store.
stateful_agent = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="chat_history",
)


# ---------------------------------------------------------------------------
# Multi-Turn Interaction Test Suite
# ---------------------------------------------------------------------------

def run_tests() -> None:
    """Runs a three-turn conversation against the stateful agent.

    Turn 1 gives the agent a name and a symptom (no tool needed).
    Turn 2 relies on memory (the symptom) plus the referral and department
    tools. Turn 3 relies on memory (the name) plus the stock tool. All turns
    share one session ID so history carries over between them.
    """
    # The session_id in config picks which stored history to use. Keeping it the
    # same across all three calls is what lets the agent remember earlier turns.
    cfg = {"configurable": {"session_id": "patient_123"}}

    # Turn 1: only states a name and a symptom. No tool is needed yet.
    print("\n--- Turn 1: memory only ---")
    r1 = stateful_agent.invoke(
        {"input": "Hi, my name is Achieng. My daughter has a persistent itchy skin rash."}, config=cfg
    )
    print("\nFinal answer:", r1["output"])

    # Turn 2: "that" refers back to the rash (memory), then two tools run:
    # referral to find the specialty, and department lookup to find its location.
    print("\n--- Turn 2: memory + referral + department tools ---")
    r2 = stateful_agent.invoke(
        {"input": "Which specialist should we see for that, and where is that department?"}, config=cfg
    )
    print("\nFinal answer:", r2["output"])

    # Turn 3: recalls the name from memory and also calls the stock tool.
    print("\n--- Turn 3: memory + stock tool ---")
    r3 = stateful_agent.invoke(
        {"input": "What is my name, and do we have amoxicillin in stock?"}, config=cfg
    )
    print("\nFinal answer:", r3["output"])


# Only run the tests when this file is executed directly, not when imported.
if __name__ == "__main__":
    run_tests()