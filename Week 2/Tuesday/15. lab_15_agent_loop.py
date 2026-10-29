# lab14_agent_loop.py
# A LangChain agent whose single tool is the compliance vault. The agent reads
# the question, decides to call the tool, reads its result, and writes the answer.

import os                                      # standard library, used for environment access
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_openai import ChatOpenAI        # the OpenAI chat model wrapper
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor  # classic agent builder + run loop
from langchain_core.prompts import ChatPromptTemplate  # builds the agent's prompt with placeholders
from langchain_core.tools import tool          # the @tool decorator that turns a function into a tool

# Load the .env file (for example OPENAI_API_KEY) into the environment.
load_dotenv()


# The @tool decorator turns this function into a tool. Its name, the type hint,
# and the docstring become the schema the agent reads to decide when to call it.
# The docstring says "ONLY for compliance-policy questions" so the agent routes
# policy questions here instead of answering from its own training.
@tool
def query_compliance_vault(question: str) -> str:
    """Use ONLY for AfyaPlus compliance-policy questions. Queries the grounded knowledge base."""
    # Mocked for this loop demo: it returns a fixed policy string. In the full
    # hybrid build this line would call the LlamaIndex query engine instead, so
    # the answer would be retrieved from your documents rather than hardcoded.
    return "Policy: procurement above $500 requires director sign-off."


# Create the reasoning model. temperature=0.0 keeps tool choice consistent.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# The agent prompt. The agent_scratchpad placeholder is required: it is where
# the agent records its tool calls and their results during the reason-act loop.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AfyaPlus assistant. Use the compliance vault for policy questions."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# Register the tool, build the agent (model + tools + prompt), then wrap it in an
# AgentExecutor. The executor runs the loop: the agent decides, the tool runs,
# the result goes back to the agent, until a final answer. verbose=True prints each step.
tools = [query_compliance_vault]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Run the agent on a policy question, wrapped in try/except so any real failure
# (bad key, network error, rate limit) prints a readable message and traceback
# instead of crashing.
try:
    result = executor.invoke({"input": "Do I need sign-off to buy a $900 centrifuge?"})
    print(f"\n--- Final Answer ---")
    # The executor returns a dict; the answer is under "output". .get provides a
    # fallback string in case that key is missing.
    print(result.get("output", "No output returned"))
except Exception as e:
    print(f"Error running agent: {e}")
    import traceback
    traceback.print_exc()