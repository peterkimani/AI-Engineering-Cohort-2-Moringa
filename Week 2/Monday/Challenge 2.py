import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

llm_triage = ChatOpenAI(model = "gpt-5.4-mini", temperature = 0.2)

prompt_explanation = ChatPromptTemplate.from_template(
    "You are an afyaplus health assistant"
    "provide a brief, professional triage explanation"
    "of the following condition for an internal medical note: "
    "\n Condition: {condition}"
)

prompt_precaution = ChatPromptTemplate.from_template(
    "Based on this triage note, append an explicit medication safety caution "
    " or warning regarding self medication"
    "\n Triage Note: {triage_note}"
)

prompt_translation = ChatPromptTemplate.from_template(
    "Combine the follownig clinical note and safety warning into a single cohesive"
    "professional SMS message translated into clear Swahili for the patient"
    "\n Note: {note}\n warning:{warning}"
)

parser = StrOutputParser()


chain_explanation = prompt_explanation | llm_triage | parser
chain_precaution = prompt_precaution |llm_triage| parser
chain_translation = prompt_translation|llm_triage| parser

target_condition = "High Fever accompanied by severe chills (suspected Malaria)"
print(f"Target_contidion: {target_condition}\n")
explanation_output = chain_explanation.invoke({"condition":{target_condition}})

print(f"Explanation: {explanation_output}\n")

precaution_output  = chain_precaution.invoke({"triage_note":explanation_output})

print(f"Precaution: {precaution_output}")

final_swahili_sms = chain_translation.invoke(
    {
        "note": explanation_output,
        "warning": precaution_output
    }
)

final_swahili_sms = chain_translation.invoke(
    {
        "note": explanation_output,
        "warning": precaution_output
    }
)
print(f"Final Swahili SMS: {final_swahili_sms}")
