# lab5_custom_tools.py

import os                                      # standard library, used for environment access
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_core.tools import tool          # the @tool decorator that turns a function into a tool
from langchain_openai import ChatOpenAI        # the OpenAI chat model wrapper

# Load the .env file (for example OPENAI_API_KEY) into the environment so the client finds the key.
load_dotenv()

# The @tool decorator wraps this function as a LangChain tool. The function
# name becomes the tool name, the type hints become its input schema, and the
# docstring becomes the description the model reads to decide when to call it.
@tool
def get_clinic_stock_count(medication_name: str) -> str:
    """Queries the AfyaPlus central inventory system for a specific drug.
    Use this tool whenever a patient asks if a medicine is physically available in the pharmacy.
    """
    # Mock stand-in for a real database. In production this lookup would hit
    # the actual inventory system instead of a hardcoded dictionary.
    inventory_db = {
        "paracetamol": "1,200 tablets available in Nairobi Hub.",
        "amoxicillin": "0 units left - currently backordered.",
        "antacid": "450 bottles in stock in Mombasa clinic."
    }
    # Lowercase and strip the input so "Amoxicillin " matches the "amoxicillin" key.
    normalized_name = medication_name.lower().strip()
    # Return the matching entry, or a not-found message if the key is absent.
    return inventory_db.get(normalized_name, f"Medication '{medication_name}' not found in database records.")

# Inspect the tool's metadata. These are the fields the model reads at runtime
# to understand what the tool is and when to use it.
print("--- Evaluating Tool Schema Metadata ---")
print(f"Tool Schema Name: {get_clinic_stock_count.name}")              # the tool's name (the function name)
print(f"Tool Schema Description: {get_clinic_stock_count.description}") # the docstring, shown to the model
# Call the tool directly to confirm it works on its own, without any model involved.
print(f"Direct Execution Output: {get_clinic_stock_count.invoke('amoxicillin')}\n")

# Create the model and attach the tool. bind_tools tells the model the tool
# exists and gives it the schema, but does NOT run the tool automatically.
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
llm_with_tools = llm.bind_tools([get_clinic_stock_count])

# Ask a question that should trigger the tool. The model does not execute the
# tool itself. Instead it returns a tool_calls list describing which tool it
# wants to run and with what arguments. Actually running it would be your job.
response = llm_with_tools.invoke("Do we have any amoxicillin in stock right now?")
print("--- Inspecting Model's Tool Call Decision ---")
if response.tool_calls:
    # The model chose to call a tool. Print each requested name and its arguments.
    for call in response.tool_calls:
        print(f"Tool requested: {call['name']}, Arguments: {call['args']}")
else:
    # The model decided no tool was needed and answered in plain text instead.
    print(f"No tool call made. Model answered directly: {response.content}")