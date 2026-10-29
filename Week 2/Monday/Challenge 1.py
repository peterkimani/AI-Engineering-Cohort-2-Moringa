import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()

llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.3)

messages = [
    SystemMessage(content = (
        "You are a triage assistant at Afyaplus health "
        "Asses the language of the human message and determine if it is either English, Sheng or swahili "
        "Respond to the message in standard and clear english "
        "Append a localised swahili clossing reasurance sentence: Afyaplus Iko hapa kukusaidia "
        "Never Diagnose or prescribe or recommend any medication or treatment to the patient, refer them to the physician "

    )),
    HumanMessage(content = "I am having a headache, recommend which painkillers to take?"),
]

response = llm.invoke(messages)
print(f"AI: {response.content}\n")