# lab9_math_suite.py

import os                                      # standard library, used for environment access
import traceback                               # formats exception details into readable text
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_core.tools import tool          # the @tool decorator that turns a function into a tool
from langchain_openai import ChatOpenAI        # the OpenAI chat model wrapper
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor  # classic agent builder + run loop
from langchain_core.prompts import ChatPromptTemplate  # builds the agent's prompt with placeholders

# Load the .env file (for example OPENAI_API_KEY) into the environment so the client finds the key.
load_dotenv()


# The @tool decorator turns this function into a tool. The function name,
# the int type hints, and this docstring become the schema the agent reads
# to decide when to call it and what arguments to pass.
@tool
def calculate_facility_utilization(active_beds: int, total_beds: int) -> str:
    """Calculates the capacity utilisation percentage for a clinic ward.

    How it works: divides active_beds by total_beds and multiplies by 100 to
    get a percentage, formatted to one decimal place. Guards against a
    divide-by-zero by returning an error string when total_beds is not
    greater than zero. Any other failure is caught and returned as an error
    string so the agent gets a message back instead of crashing.
    """
    try:
        # Guard clause: utilisation is undefined if there are no beds to divide by.
        if total_beds <= 0:
            return "Error: total beds must be greater than zero."
        if active_beds > total_beds:
            return "Error: active beds cannot exceed total beds."
        # Core calculation: fraction occupied, expressed as a percentage.
        pct = (active_beds / total_beds) * 100
        # :.1f formats the number to one decimal place, e.g. 84.0.
        return f"Utilisation: {pct:.1f}%"
    except Exception:
        # Return the error as text (limit=1 keeps it to the last frame) so the
        # agent receives a usable message rather than an unhandled exception.
        return f"Error computing utilisation: {traceback.format_exc(limit=1)}"


# Create the reasoning model. temperature=0.0 keeps tool choice consistent.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# The agent prompt. The agent_scratchpad placeholder is required: it is where
# the agent records its tool calls and their results during the reason-act loop.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AfyaPlus operations assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Register the tool(s), build the agent (model + tools + prompt), then wrap it
# in an AgentExecutor. The executor runs the loop: the agent decides, the tool
# runs, the result goes back to the agent, until a final answer. verbose=True
# prints each step.
tools = [calculate_facility_utilization]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent on one question, wrapped in try/except so any failure prints
# a readable message and full traceback instead of a bare crash.
try:
    result = executor.invoke({"input": "If 84 of our 80 beds are occupied, what is utilisation?"})
    print(f"\n--- Final Answer ---")
    # The executor returns a dict; the answer is under "output". .get provides
    # a fallback string in case that key is missing.
    print(result.get("output", "No output returned"))
except Exception as e:

    print(f"Error running agent: {e}")
    # Print the full traceback to help debug what went wrong.
    #traceback.print_exc()