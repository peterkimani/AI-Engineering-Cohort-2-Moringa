# challenge4_diagnostic_referral_agent.py
# Challenge 4: Engineering a Specialized Diagnostic & Referral Agent
#
# Requires: langchain-classic, langchain-openai, python-dotenv

import os                                      # standard library, used for environment access
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_openai import ChatOpenAI        # the OpenAI chat model wrapper
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor  # agent builder + run loop
from langchain_core.prompts import ChatPromptTemplate  # builds the agent's prompt with placeholders
from langchain_core.tools import tool          # the @tool decorator that turns a function into a tool

# Load the .env file (for example OPENAI_API_KEY) into the environment so the client finds the key.
load_dotenv()


# TOOL 1: stock lookup. The @tool decorator turns this function into a tool.
# The docstring is what the agent reads to decide when to pick this tool, so
# it says ONLY use it for stock and NOT for referrals, to avoid confusion.
@tool
def get_clinic_stock_count(medication_name: str) -> str:
    """Use this tool ONLY to check how many units of a medication are in stock
    at the AfyaPlus clinic pharmacy. Input is a medication name such as
    'amoxicillin' or 'paracetamol'. Do not use it for referrals.

    How it works: the name is lowercased and stripped of surrounding spaces,
    then looked up in a hardcoded stock dictionary. If the medication is not
    listed, the count defaults to 0. The result is returned as a string such
    as "120 units".
    """
    # Mock inventory. A real system would query a database here instead.
    stock = {"amoxicillin": 120, "paracetamol": 540, "ibuprofen": 75}
    # Normalise the input (lowercase + strip) so "Paracetamol " matches the key.
    # .get(..., 0) returns 0 when the medication is not in the dictionary.
    count = stock.get(medication_name.strip().lower(), 0)
    return f"{count} units"


# TOOL 2: specialist referral. Its docstring deliberately contrasts with the
# stock tool so the agent routes symptom questions here and stock questions
# to the other tool.
@tool
def refer_to_specialist(symptoms: str) -> str:
    """Use this tool ONLY when a patient describes symptoms or a condition and
    you need to decide WHICH medical specialist they should be referred to.
    Input is a short free-text description of the symptoms. Do not use it for
    medication stock.

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
    # Pick the specialty with the highest score. key=scores.get makes max()
    # compare the dictionary's values (the counts), returning the top key.
    best_specialty = max(scores, key=scores.get)

    # A top score of 0 means no keyword matched, so fall back to a GP review.
    if scores[best_specialty] == 0:
        return "No clear specialist match. Recommend a general practitioner review first."
    return f"Refer the patient to {best_specialty}."


# Create the reasoning model. temperature=0.0 keeps tool choice consistent.
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.0)

# Bind both tools so the agent can choose between them per query
tools = [get_clinic_stock_count, refer_to_specialist]

# The agent prompt. The agent_scratchpad placeholder is required: it is where
# the agent records its tool calls and their results during the reason-act loop.
prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You are an AfyaPlus diagnostic and referral assistant. "
     "Pick the right tool for each question and explain the result clearly."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}"),
])

# Build the agent (model + tools + prompt), then wrap it in an AgentExecutor.
# The executor runs the loop: the agent decides, a tool runs, the result goes
# back to the agent, and so on until a final answer. verbose=True prints each step.
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


def run_tests() -> None:
    """Runs one stock query and one referral query against the executor.

    Query 1 should trigger only get_clinic_stock_count. Query 2 should
    trigger only refer_to_specialist. The final answer of each run is read
    from the "output" key of the executor response and printed.
    """
    # Test 1: a stock question. The agent should pick the stock tool.
    print("\n--- Test 1: stock query (expects get_clinic_stock_count) ---")
    r1 = executor.invoke({"input": "How much paracetamol do we have in stock?"})
    print("\nFinal answer:", r1["output"])  # the executor returns the answer under "output"

    # Test 2: a symptom question. The agent should pick the referral tool.
    print("\n--- Test 2: referral query (expects refer_to_specialist) ---")
    r2 = executor.invoke({
        "input": "A patient has chest pain and shortness of breath. Who should they be referred to?"
    })
    print("\nFinal answer:", r2["output"])


# Only run the tests when this file is executed directly, not when imported.
if __name__ == "__main__":
    run_tests()