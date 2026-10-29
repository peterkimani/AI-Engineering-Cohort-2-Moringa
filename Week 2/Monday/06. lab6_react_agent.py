# lab6_react_agent.py
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
#from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

load_dotenv()

@tool
def get_clinic_stock_count(medication_name: str) -> str:
    """Returns current stock for a medication at the AfyaPlus clinic."""
    stock = {"amoxicillin": 120, "paracetamol": 540}
    return f"{stock.get(medication_name.lower(), 0)} units"

llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.0)
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an AfyaPlus operations assistant."),
    ("human", "{input}"),
    ("placeholder", "{agent_scratchpad}")
])

tools = [get_clinic_stock_count]
agent = create_tool_calling_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

executor.invoke({"input": "Do we have enough piriton in stock?"})