import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm_router = ChatOpenAI(model="gpt-5.4-mini", temperature=0.0)
llm_chain = ChatOpenAI(model = "gpt-5.4-mini",temperature=0.2)

parser = StrOutputParser()

#Task 1 
router_template = (
"Classifty the following patient message into either 'Clinical' or 'Biling'. "
"Return only the word critical or billing"

"Message: {message}"
)

router_prompt = ChatPromptTemplate.from_template(router_template)

clinical_template = ("Draft a 1- sentence medical triage acknowledgement for this message. "
"Message: {message}")

billing_template = (
    "Draft a sentence billing support acknowledgement for this message. "
    "Message: {message}"
)

translation_template = (
    "Translate this text into clear Swahili for an SMS alert: {text} "
)
translation_prompt = ChatPromptTemplate.from_template(translation_template)

router_chain = router_prompt | llm_router | parser
translation_chain = translation_prompt |llm_chain | parser

incoming_patient_sms = (
    "I am having a headache."
)

print(f"Incoming Payload:{incoming_patient_sms}\n")

detect_route = router_chain.invoke(
    {"message":incoming_patient_sms}
).strip().upper()

print(f"--- Route Evaluated: {detect_route}---")

if "CLINICAL" in detect_route:
    draft_prompt = ChatPromptTemplate.from_template(clinical_template)
    draft_chain = draft_prompt |llm_chain| parser
    english_draft = draft_chain.invoke({"message":incoming_patient_sms})
elif "BILLING" in detect_route:
    draft_prompt = ChatPromptTemplate.from_template(billing_template)
    draft_chain = draft_prompt |llm_chain| parser
    english_draft = draft_chain.invoke({"message":incoming_patient_sms})
else:
    raise ValueError("Routing Failed to identify a category")


swahili_sms = translation_chain.invoke({"text":english_draft})

print(f"Draft english: {english_draft}")

print(f"Final Swahili SMS: {swahili_sms}")