# challenge_supplier_coordination.py
# Core skill (from Lab 5): a lookup tool whose docstring makes the agent
# call it at the right time. Two tools chain to route a procurement request.

import os                                      # standard library, used for environment access
import traceback                               # formats exception details into readable text
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_openai import ChatOpenAI        # the OpenAI chat model wrapper
from langchain_core.tools import tool          # the @tool decorator that turns a function into a tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor  # classic agent builder + run loop
from langchain_core.prompts import ChatPromptTemplate  # builds the agent's prompt with placeholders

# Load the .env file (for example OPENAI_API_KEY) into the environment so the client finds the key.
load_dotenv()


# ---------------------------------------------------------------------------
# Step 1: route an item category to its storage facility and section
# ---------------------------------------------------------------------------

# The docstring tells the agent this tool is the FIRST step: it takes a type of
# item and returns which depot and section it belongs to.
@tool
def determine_storage_facility(item_category: str) -> str:
    """Use this tool to find which storage facility and section an incoming
    item category should be routed to. Input is an item category such as
    'cold-chain items' or 'general supplies'. Call this first, before looking
    up a representative, because the representative depends on the section.

    How it works: the category is lowercased and checked for the phrase
    'cold-chain'. Cold-chain items need refrigeration, so they route to the
    Refrigerated Depot (Section X). Everything else routes to the General
    Inventory Depot (Section Y).
    """
    # Lowercase so "Cold-Chain Items" matches regardless of how it was typed.
    if "cold-chain" in item_category.lower():
        return "Refrigerated Depot - Section X"
    return "General Inventory Depot - Section Y"


# ---------------------------------------------------------------------------
# Step 2: route a section to its inventory representative
# ---------------------------------------------------------------------------

# The docstring tells the agent this is the SECOND step: it takes a section
# (the output of the first tool) and returns the person responsible for it.
@tool
def lookup_inventory_representative(facility_section: str) -> str:
    """Use this tool to find the inventory representative responsible for a
    given facility section. Input is a facility or section description such as
    'Section X' or 'Refrigerated Depot - Section Y'. Call this after
    determine_storage_facility, using the section it returned.

    How it works: the input is lowercased and scanned for 'section x' or
    'section y'. Section X is covered by Representative Mwangi and Section Y by
    Representative Otieno. If neither section is found, a clean not-found
    message is returned.
    """
    text = facility_section.lower()  # lowercase once for case-insensitive matching
    # Match whichever section appears in the text and return its representative.
    if "section x" in text:
        return "Representative Mwangi (mwangi@afyaplus.co)"
    if "section y" in text:
        return "Representative Otieno (otieno@afyaplus.co)"
    return "No representative found for that section."


# Create the reasoning model. temperature=0.0 keeps tool choice consistent.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)

# The agent prompt. The agent_scratchpad placeholder is required: it is where
# the agent records its tool calls and their results during the reason-act loop.
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AfyaPlus supplier coordination assistant. Route each "
                "request to the correct facility, then to the representative for "
                "that facility, and present the contact clearly."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

# ---------------------------------------------------------------------------
# Step 3: wire the tools into an agent and run the compound query
# ---------------------------------------------------------------------------

# Register both tools. Their docstrings describe a clear order (facility first,
# representative second), so the agent chains them: the section returned by the
# first tool becomes the input to the second. verbose=True logs every step.
tools = [determine_storage_facility, lookup_inventory_representative]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)


# Only run the test when this file is executed directly, not when imported.
if __name__ == "__main__":
    # This query names the item category but not the section or the person, so
    # the agent must call the first tool to get the section, then the second to
    # get the representative. That is the chaining this challenge is testing.
    query = (
        "We have a shipment of cold-chain items arriving. Who is the active "
        "inventory representative we should contact at their assigned facility?"
    )
    try:
        res = executor.invoke({"input": query})
        print(f"\nFinal System Resolution:\n{res.get('output', 'No output returned')}")
    except Exception as e:
        print(f"Error running agent: {e}")
        traceback.print_exc()