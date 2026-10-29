# lab2_4_2_tool_binding.py
# Wraps the LlamaIndex query engine from the previous lab as a LangChain @tool,
# so a LangChain agent can call your RAG pipeline as one of its tools.

import os                                      # standard library, used for environment access
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_core.tools import tool          # the @tool decorator that turns a function into a tool

# Import the already-built query engine from the ingestion lab. Importing it runs
# that file top to bottom, so the documents are loaded, split, embedded, and
# indexed again now, and base_query_engine is ready to answer queries here.
from lab_12_llamaIndex import base_query_engine

# Load the .env file (for example OPENAI_API_KEY) into the environment.
load_dotenv()

print("--- Step 4: Binding LlamaIndex Query Engine to LangChain Interface ---")


# The @tool decorator turns this function into a LangChain tool. Its name, the
# type hint, and this docstring become the schema an agent reads to decide when
# to call it. The docstring deliberately lists the topics it covers (regulations,
# financial limits, allowances, clinic metadata) so the agent routes those
# questions here rather than guessing an answer itself.
@tool
def query_internal_compliance_vault(user_search_query: str) -> str:
    """Queries the internal AfyaPlus regional compliance protocols database.
    Use this tool whenever the user asks questions regarding company regulations,
    employee financial limits, medical allowances, or operational clinic metadata.
    """
    # try/except so a failure returns a readable string instead of crashing the
    # agent. A tool the agent calls automatically must not raise.
    try:
        # Hand the tool's input straight to the LlamaIndex engine, which runs the
        # full RAG flow (embed the question, retrieve top nodes, answer from them).
        engine_response = base_query_engine.query(user_search_query)
        # The engine returns a response object; str() turns it into plain text,
        # which is the type a LangChain tool must return.
        return str(engine_response)
    except Exception as error:
        # Return the error as text so the agent receives a usable message.
        return f"Database Retrieval failure encountered: {str(error)}"


# Run only when this file is executed directly, not when imported.
if __name__ == "__main__":
    print("\nExecuting Tool Invocations Directly via LangChain Interface...")
    # Call the tool directly (no agent yet) to confirm the wrapper works. The
    # input is a dict whose key matches the function's parameter name,
    # user_search_query, because that is how a LangChain tool receives arguments.
    tool_payload = {"user_search_query": "Nairobi hub outpatient medical allowance limits"}
    # .invoke runs the tool with that input and returns its string result.
    result = query_internal_compliance_vault.invoke(tool_payload)
    print(f"\nWrapped Tool Result Output:\n{result}")