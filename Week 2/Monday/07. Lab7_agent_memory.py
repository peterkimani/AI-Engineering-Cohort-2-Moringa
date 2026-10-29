# lab7_agent_memory.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory

load_dotenv()

@tool
def get_clinic_stock_count(medication_name: str) -> str:
    """Returns current stock for a medication at the AfyaPlus clinic.

    Looks up the lowercased medication name in a hardcoded dictionary of
    stock levels (currently only amoxicillin: 120). If the name is not
    found, the default of 0 is used. The count is returned as a string
    such as "120 units".
    """
    return f"{ {'amoxicillin': 120}.get(medication_name.lower(), 0) } units"

llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AfyaPlus assistant. Remember the patient across turns."),
    ("placeholder", "{chat_history}"),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

tools = [get_clinic_stock_count]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

store = {}

def get_session_history(session_id: str):
    """Returns the chat history object for a given session ID.

    Uses the module-level `store` dict as an in-memory lookup keyed by
    session_id. If no history exists for that ID yet, a new empty
    ChatMessageHistory is created and saved. The same object is returned
    on later calls, so messages accumulate across turns. Nothing is
    persisted to disk, so history is lost when the script exits.
    """
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

agent_with_memory = RunnableWithMessageHistory(
    executor, get_session_history,
    input_messages_key="input", history_messages_key="chat_history"
)

cfg = {"configurable": {"session_id": "patient-001"}}
agent_with_memory.invoke({"input": "My name is Achieng."}, cfg)
result = agent_with_memory.invoke({"input": "What is my name, and do we have amoxicillin?"}, cfg)
print(result["output"])