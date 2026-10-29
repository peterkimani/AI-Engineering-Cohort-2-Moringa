# challenge8_depreciation.py
# Core skill (from Lab 9): a validated @tool wired into an AgentExecutor.

import os                                      # standard library, used for environment access
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_openai import ChatOpenAI        # the OpenAI chat model wrapper
from langchain_core.tools import tool          # the @tool decorator that turns a function into a tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor  # classic agent builder + run loop
from langchain_core.prompts import ChatPromptTemplate  # builds the agent's prompt with placeholders

# Load the .env file (for example OPENAI_API_KEY) into the environment so the client finds the key.
load_dotenv()


# The @tool decorator turns this function into a tool. Its name, the type hints,
# and this docstring become the schema the agent reads to decide when to call it.
@tool
def calculate_asset_depreciation(initial_cost: float, salvage_value: float, useful_life_years: int) -> str:
    """Calculates annual straight-line depreciation for a clinic asset.

    How it works: straight-line depreciation spreads an asset's loss in value
    evenly across its useful life. The formula is
    (initial_cost - salvage_value) / useful_life_years, which gives the amount
    the asset depreciates each year. The result is formatted to two decimals.

    Validation: useful_life_years is the denominator, so a value of zero or
    below would cause a division-by-zero crash or a nonsense result. The guard
    clause checks for that first and returns a clean error string instead,
    so the agent receives a usable message rather than an exception.
    """
    # Guard the denominator before dividing: a zero or negative life is invalid.
    if useful_life_years <= 0:
        return "Error: useful life must be positive; division by zero is prohibited."
    # Core calculation: depreciable amount spread evenly over the asset's life.
    annual = (initial_cost - salvage_value) / useful_life_years
    # :.2f formats the figure to two decimal places, e.g. 900.00.
    return f"Annual depreciation: {annual:.2f}"


# Create the reasoning model. temperature=0.0 keeps tool choice consistent.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# The agent prompt. The agent_scratchpad placeholder is required: it is where
# the agent records its tool calls and their results during the reason-act loop.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AfyaPlus logistics assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Register the tool, build the agent (model + tools + prompt), then wrap it in an
# AgentExecutor. The executor runs the loop: the agent decides, the tool runs,
# the result goes back to the agent, until a final answer. verbose=True prints each step.
tools = [calculate_asset_depreciation]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent on a depreciation question, wrapped in try/except so any real
# failure (bad key, network error, rate limit) prints a readable message and
# traceback instead of crashing.
try:
    result = executor.invoke({"input": "Yearly depreciation for a $5,000 server, $500 salvage, 5-year life?"})
    print(f"\n--- Final Answer ---")
    # The executor returns a dict; the answer is under "output". .get provides a
    # fallback string in case that key is missing.
    print(result.get("output", "No output returned"))
except Exception as e:
    print(f"Error running agent: {e}")
    import traceback
    traceback.print_exc()