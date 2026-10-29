# challenge7_routed_translation_pipeline.py
# Challenge 7: Architectural Design Blueprinting
#
# Runnable implementation of the blueprint below: a patient message is routed,
# then (if safe) sent through a sequential acknowledgement-plus-translation chain.
#
# Requires: langchain-openai, python-dotenv
#
# ===========================================================================
# ARCHITECTURE BLUEPRINT
# ===========================================================================
#
#   [1] PATIENT MESSAGE (input)
#            |
#            v
#   [2] SAFETY ROUTER  ............ DETERMINISTIC (LLM, temperature 0.0)
#       Classifies: INFO or EMERGENCY
#       GUARDRAIL A: fails closed. Anything that is not exactly "INFO"
#                    is treated as EMERGENCY.
#            |
#            v
#   [3] BRANCH LOGIC (if / else) .. DETERMINISTIC (plain Python)
#            |                                   |
#       label == EMERGENCY                  label == INFO
#            |                                   |
#            v                                   v
#   [4a] FIXED EMERGENCY REPLY         [4b] SEQUENTIAL CHAIN ..... DYNAMIC (LLM)
#        Hardcoded text, English            |
#        and Swahili. No LLM.               v
#        GUARDRAIL B: bypasses         Step 1: ACKNOWLEDGEMENT
#        the sequential chain               LLM writes a short, empathetic
#        entirely.                          English reply. No diagnosis.
#            |                              GUARDRAIL C: prompt forbids
#            |                              diagnosis and treatment advice.
#            |                              |
#            |                              v (Step 1 output feeds Step 2)
#            |                         Step 2: TRANSLATION
#            |                              LLM translates the reply to Swahili.
#            |                              GUARDRAIL D: instructed to keep
#            |                              numbers, names and meaning unchanged.
#            |                              |
#            +---------------+--------------+
#                            |
#                            v
#   [5] OUTPUT: route label + English reply + Swahili reply
#
# Deterministic logic : [2] router (temperature 0), [3] branch, [4a] fixed reply
# Dynamic logic       : [4b] acknowledgement and translation steps only
# ===========================================================================

import os                                      # standard library, used for environment access
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_openai import ChatOpenAI        # the OpenAI chat model wrapper
from langchain_core.prompts import ChatPromptTemplate  # builds prompts with placeholders
from langchain_core.output_parsers import StrOutputParser  # turns the model's message into a plain string
from langchain_core.runnables import RunnableParallel, RunnableLambda  # compose steps into a runnable chain

# Load the .env file (for example OPENAI_API_KEY) into the environment so the client finds the key.
load_dotenv()

MODEL_NAME = "gpt-5.4-mini"  # one place to change the model name for the whole file
TARGET_LANGUAGE = "Swahili"  # the language the acknowledgement is translated into

# Hardcoded emergency text, written in advance in both languages (no LLM used).
# Using fixed text means the emergency reply cannot be changed or delayed by a model.
FIXED_EMERGENCY_EN = (
    "This may be a medical emergency. Call 999 or 112 now, or go to the nearest "
    "emergency department immediately."
)
FIXED_EMERGENCY_SW = (
    "Hii inaweza kuwa dharura ya kiafya. Piga simu 999 au 112 sasa, au nenda mara moja "
    "kwenye idara ya dharura iliyo karibu nawe."
)


# ---------------------------------------------------------------------------
# [2] Safety router (deterministic)
# ---------------------------------------------------------------------------

# temperature=0.0 makes the classification repeatable, which is the whole point
# of a safety gate: the same message always routes the same way.
router_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
# Prompt that defines both labels and asks for a single word. "If unsure, answer
# EMERGENCY" biases the gate toward safety on ambiguous input.
router_prompt = ChatPromptTemplate.from_template(
    "You are a triage safety classifier for a clinic assistant.\n"
    "Classify the patient message as exactly one word:\n"
    "EMERGENCY: life-threatening or urgent situation (chest pain, trouble breathing, "
    "severe bleeding, stroke signs, seizure, overdose, unconsciousness, suicidal intent).\n"
    "INFO: general, non-urgent questions or requests.\n"
    "If unsure, answer EMERGENCY.\n"
    "Reply with only the single word INFO or EMERGENCY.\n\n"
    "Message: {msg}\nAnswer:"
)
# The router chain: fill the prompt, send it to the model, parse the reply to a string.
router_chain = router_prompt | router_llm | StrOutputParser()


def classify_message(message: str) -> str:
    """Classifies a patient message as INFO or EMERGENCY.

    Runs the zero-temperature router chain, strips whitespace and uppercases
    the output. Only an exact "INFO" label returns INFO. Any other output is
    mapped to EMERGENCY, so an unclear classification fails closed and never
    reaches the dynamic chain.
    """
    # Run the router and normalise the output so the comparison below is reliable.
    label = router_chain.invoke({"msg": message}).strip().upper()
    # Fail closed (GUARDRAIL A): only an exact "INFO" passes; everything else is EMERGENCY.
    return "INFO" if label == "INFO" else "EMERGENCY"


# ---------------------------------------------------------------------------
# [4b] Sequential chain: acknowledgement -> translation (dynamic)
# ---------------------------------------------------------------------------

# A little temperature here (0.3) lets the acknowledgement read naturally,
# unlike the router which must be fully deterministic.
chain_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.3)

# Step 1 prompt: write the English acknowledgement. The last line is GUARDRAIL C,
# forbidding diagnosis or treatment advice.
ack_prompt = ChatPromptTemplate.from_template(
    "You are a courteous AfyaPlus clinic assistant. Write a short reply (2 to 3 sentences) "
    "that acknowledges the patient's message and tells them how the clinic can help. "
    "Do not give a diagnosis or treatment advice.\n\n"
    "Patient message: {msg}\nReply:"
)
ack_chain = ack_prompt | chain_llm | StrOutputParser()

# Step 2 prompt: translate the English reply. "Keep numbers, names and meaning
# unchanged" is GUARDRAIL D. {acknowledgement} is filled with Step 1's output.
translate_prompt = ChatPromptTemplate.from_template(
    "Translate the following clinic reply into " + TARGET_LANGUAGE + ". "
    "Keep numbers, names and meaning unchanged. Return only the translation.\n\n"
    "Reply: {acknowledgement}\nTranslation:"
)
translate_chain = translate_prompt | chain_llm | StrOutputParser()

# Sequential composition, read top to bottom:
#   RunnableParallel(acknowledgement=ack_chain)
#       runs Step 1 and stores its text under the key "acknowledgement".
#   The second RunnableParallel then receives that dict and produces two keys:
#       english    : copies the acknowledgement through unchanged (RunnableLambda
#                    just pulls out the "acknowledgement" value).
#       translated : feeds the same dict into translate_chain, whose prompt reads
#                    {acknowledgement}, giving the Swahili translation.
# The two dicts are joined with | so the whole thing is one runnable with .invoke().
sequential_chain = (
    RunnableParallel(acknowledgement=ack_chain)
    | RunnableParallel(
        english=RunnableLambda(lambda x: x["acknowledgement"]),
        translated=translate_chain,
    )
)


# ---------------------------------------------------------------------------
# [3] Branch logic and [5] output
# ---------------------------------------------------------------------------

def handle_patient_message(message: str) -> dict:
    """Runs the full pipeline for one patient message.

    Calls classify_message to get the route. If the route is EMERGENCY, it
    returns the hardcoded English and Swahili texts directly, so the
    sequential chain is never called. If the route is INFO, it runs the
    sequential chain, whose first step writes the English acknowledgement and
    whose second step translates it. The result is a dict holding the route
    label, the English reply and the translated reply.
    """
    route = classify_message(message)  # [2] run the safety router

    # [4a] EMERGENCY path: return fixed text and stop. The return here is
    # GUARDRAIL B: it means the sequential chain below is never reached.
    if route == "EMERGENCY":
        return {
            "route": "EMERGENCY (deterministic, hardcoded)",
            "english": FIXED_EMERGENCY_EN,
            "translated": FIXED_EMERGENCY_SW,
        }

    # [4b] INFO path: run the two-step chain (acknowledge, then translate).
    result = sequential_chain.invoke({"msg": message})
    # [5] assemble the output dict from the chain's two keys.
    return {
        "route": "INFO (sequential chain: acknowledge, then translate)",
        "english": result["english"],
        "translated": result["translated"],
    }


def run_tests() -> None:
    """Runs one INFO message and one EMERGENCY message through the pipeline.

    Prints the route label, the English reply and the translated reply for
    each, so you can confirm the INFO message went through both chain steps
    and the emergency message used the fixed path.
    """
    # First message should route INFO, the second should route EMERGENCY.
    tests = [
        "Hello, I would like to know how to book a check-up for my mother next week.",
        "My husband has severe chest pain and is struggling to breathe.",
    ]
    for msg in tests:
        print("\n" + "=" * 60)
        print("Message:", msg)
        out = handle_patient_message(msg)         # run the full pipeline
        print("Route  :", out["route"])           # which path was taken
        print("English:", out["english"])         # English reply
        print(f"{TARGET_LANGUAGE}:", out["translated"])  # translated reply


# Only run the tests when this file is executed directly, not when imported.
if __name__ == "__main__":
    run_tests()