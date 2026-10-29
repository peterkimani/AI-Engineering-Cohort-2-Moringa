# lab10_multi_tool.py

import os                                      # standard library, used for environment access
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_core.tools import tool          # the @tool decorator that turns a function into a tool
from langchain_openai import ChatOpenAI        # the OpenAI chat model wrapper
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor  # classic agent builder + run loop
from langchain_core.prompts import ChatPromptTemplate  # builds the agent's prompt with placeholders

# Load the .env file (for example OPENAI_API_KEY) into the environment so the client finds the key.
load_dotenv()


# TOOL 1: stock lookup. The @tool decorator turns this function into a tool.
# The name, the str type hint, and this docstring become the schema the agent
# reads to decide when to call it.
@tool
def get_clinic_stock_count(medication_name: str) -> str:
    """Returns the current stock count for a medication at the AfyaPlus clinic.

    How it works: lowercases the medication name and looks it up in a hardcoded
    stock dictionary. If the name is not found, the count defaults to 0. The
    result is returned as a string such as "120 units".
    """
    stock = {"amoxicillin": 120, "paracetamol": 540}
    # .get(..., 0) returns the stored count, or 0 when the medication is absent.
    return f"{stock.get(medication_name.lower(), 0)} units"


# TOOL 2: shift cost. A second, distinct tool so the agent has a choice to make.
@tool
def calculate_shift_cost(hours: float, hourly_rate: float) -> str:
    """Calculates the total cost of a staff shift.

    How it works: multiplies hours by hourly_rate and formats the result to two
    decimal places. Guards against nonsense input by returning an error string
    if either value is negative.
    """
    # Guard clause: neither hours nor rate can be negative, so reject that early.
    if hours < 0 or hourly_rate < 0:
        return "Error: hours and rate must be non-negative."
    # Core calculation: total pay = hours worked x rate. :.2f gives two decimals.
    return f"Shift cost: {hours * hourly_rate:.2f}"


# Create the reasoning model. temperature=0.0 keeps tool choice consistent.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# The agent prompt. The agent_scratchpad placeholder is required: it is where
# the agent records its tool calls and their results during the reason-act loop.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AfyaPlus operations assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Register both tools, build the agent (model + tools + prompt), then wrap it in
# an AgentExecutor. With two tools available the agent decides which to call for
# each part of the question. A compound question can trigger both in turn.
# verbose=True prints each step of the loop.
tools = [get_clinic_stock_count, calculate_shift_cost]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent on a compound question, wrapped in try/except so any real
# failure (bad key, network error, rate limit) prints a readable message and
# traceback instead of crashing.
try:
    result = executor.invoke({"input": "How much paracetamol do we have, and what is the cost of an 8-hour shift at KES 450 per hour?"})
    print(f"\n--- Final Answer ---")
    # The executor returns a dict; the answer is under "output". .get provides a
    # fallback string in case that key is missing.
    print(result.get("output", "No output returned"))
except Exception as e:
    print(f"Error running agent: {e}")
    import traceback
    traceback.print_exc()