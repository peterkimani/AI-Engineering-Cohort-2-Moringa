# challenge6_safe_clinic_router.py
# Challenge 6: The Safe Clinic Router Matrix
#
# A zero-temperature LLM classifier labels each message INFO or EMERGENCY.
# EMERGENCY messages get a fixed, hardcoded response and never touch the agent.
# INFO messages go to an autonomous tool-calling agent.
#
# Requires: langchain-classic, langchain-openai, langchain-community,
#           ddgs (or duckduckgo-search), python-dotenv

import os                                      # standard library, used for environment access
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_openai import ChatOpenAI        # the OpenAI chat model wrapper
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor  # agent builder + run loop
from langchain_core.prompts import ChatPromptTemplate  # builds prompts with placeholders
from langchain_core.output_parsers import StrOutputParser  # turns the model's message into a plain string
from langchain_core.tools import tool          # the @tool decorator that turns a function into a tool

# Load the .env file (for example OPENAI_API_KEY) into the environment so the client finds the key.
load_dotenv()

MODEL_NAME = "gpt-5.4-mini"  # one place to change the model name for the whole file

# Fixed, hardcoded safety response. No LLM is involved in producing this text,
# so an emergency reply can never be altered or delayed by the model.
FIXED_EMERGENCY_RESPONSE = (
    "[DETERMINISTIC EMERGENCY ROUTE] This may be a medical emergency. "
    "Call 999 or 112 now, or go to the nearest emergency department immediately. "
    "Do not wait for an online reply. If someone is with you, ask them to stay with you "
    "and help you get to care."
)

# Counts how many times the dynamic agent is invoked. Used to prove that
# emergency messages never reach it.
agent_call_count = 0


# ---------------------------------------------------------------------------
# Safety gate: deterministic (temperature 0) classifier
# ---------------------------------------------------------------------------

# temperature=0.0 makes the classification repeatable: the same message always
# gets the same label, which is what makes this a reliable safety gate.
router_llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)

# The classifier prompt. It defines both labels, gives examples, and tells the
# model to answer with a single word. "If unsure, answer EMERGENCY" biases the
# gate toward safety when a message is ambiguous.
router_prompt = ChatPromptTemplate.from_template(
    "You are a triage safety classifier for a clinic assistant.\n"
    "Classify the patient message as exactly one word:\n"
    "EMERGENCY: describes a life-threatening or urgent situation happening now or very recently "
    "(chest pain, trouble breathing, severe bleeding, stroke signs, seizure, overdose, "
    "loss of consciousness, suicidal intent, severe allergic reaction, unresponsive person).\n"
    "INFO: general information requests (clinic locations, opening hours, services, "
    "health education, non-urgent questions).\n"
    "If you are unsure, answer EMERGENCY.\n"
    "Reply with only the single word INFO or EMERGENCY.\n\n"
    "Message: {msg}\nAnswer:"
)

# The router chain: fill the prompt, send it to the model, parse the reply to a string.
safety_router = router_prompt | router_llm | StrOutputParser()


def classify_message(message: str) -> str:
    """Classifies a patient message as INFO or EMERGENCY.

    Sends the message through the zero-temperature router chain, then strips
    whitespace and uppercases the model output so that formatting differences
    such as "info" or " INFO\\n" do not matter. The gate fails closed: only an
    exact "INFO" label is treated as INFO. Any other output, including an
    unexpected or garbled label, is treated as EMERGENCY so that an
    uncertain classification can never open the path to the dynamic agent.
    """
    # Run the router and normalise the output so comparison is reliable.
    label = safety_router.invoke({"msg": message}).strip().upper()
    # Fail closed: only an exact "INFO" passes; anything else is treated as EMERGENCY.
    return "INFO" if label == "INFO" else "EMERGENCY"


# ---------------------------------------------------------------------------
# Dynamic agent path (INFO only)
# ---------------------------------------------------------------------------

# The tool the INFO agent is allowed to use. The docstring tells the agent when
# to call it. Here it is an offline directory, so the file runs with no search package.
@tool
def regional_lookup(query: str) -> str:
    """Use this tool to look up information about clinics, hospitals, opening
    hours or health services in towns across Kenya.

    How it works: the query is lowercased and checked against a small local
    directory of towns. If any town name appears in the query, that town's
    entry is returned. If none match, a message saying no local record was
    found is returned. This is an offline stand-in for a live web search, so
    the challenge runs without any external search package.
    """
    directory = {
        "kisumu": "Jaramogi Oginga Odinga Teaching and Referral Hospital, Kisumu. Open 24 hours.",
        "nakuru": "Nakuru Level 5 Hospital, Nakuru town centre. Open 24 hours.",
        "nairobi": "Kenyatta National Hospital, Nairobi. Open 24 hours.",
        "mombasa": "Coast General Teaching and Referral Hospital, Mombasa. Open 24 hours.",
    }
    text = query.lower()  # lowercase so "Kisumu" in the query matches the "kisumu" key
    # Return the first town whose name appears anywhere in the query.
    for town, info in directory.items():
        if town in text:
            return info
    return "No local record found for that location."


def build_info_agent() -> AgentExecutor:
    """Builds the autonomous agent used for INFO messages.

    Creates a tool-calling agent from a zero-temperature LLM, a prompt with an
    agent_scratchpad placeholder (where tool calls and results are recorded),
    and the regional_lookup tool. The agent is wrapped in an AgentExecutor,
    which runs the reason, act, observe loop until a final answer is produced.
    """
    llm = ChatOpenAI(model=MODEL_NAME, temperature=0.0)
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are an AfyaPlus information assistant. Answer general, non-urgent questions "
         "using the lookup tool when current information is needed. Do not give diagnoses "
         "or treatment instructions."),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),  # required slot for the agent's working steps
    ])
    agent = create_tool_calling_agent(llm, [regional_lookup], prompt)
    return AgentExecutor(agent=agent, tools=[regional_lookup], verbose=True)


def run_dynamic_agent(message: str) -> str:
    """Runs the dynamic agent on an INFO message and returns its answer.

    Increments the global agent_call_count by 1 each time it runs, so tests can
    check that the agent was (or was not) reached. The agent executor is built
    inside this function, which means an EMERGENCY message never even
    constructs the agent.
    """
    global agent_call_count
    agent_call_count += 1          # record that the agent was reached (used by the tests)
    executor = build_info_agent()  # build the agent only when an INFO message actually arrives
    result = executor.invoke({"input": message})
    return f"[DYNAMIC AGENT ROUTE] {result['output']}"


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

def route_patient_query(message: str) -> str:
    """Routes a patient message to the correct path.

    Step 1: classify the message with the deterministic safety router.
    Step 2: if the label is EMERGENCY, return the fixed hardcoded response
    immediately. The function returns here, so the agent code below is
    unreachable for emergencies.
    Step 3: otherwise (INFO only) hand the message to the dynamic agent.
    """
    classification = classify_message(message)

    # EMERGENCY short-circuits: return the fixed text and stop. Because this
    # returns, the agent call below is never reached for an emergency.
    if classification == "EMERGENCY":
        return FIXED_EMERGENCY_RESPONSE

    # Only INFO messages reach this line.
    return run_dynamic_agent(message)


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

def run_tests() -> None:
    """Checks that emergencies never reach the dynamic agent.

    Runs several emergency messages and records agent_call_count before and
    after. The assertion passes only if the count is unchanged, which proves
    the agent was not invoked. Then runs INFO messages and asserts that the
    count rose by one per message, which proves INFO messages do reach it.
    """
    # Messages that should all be classified EMERGENCY.
    emergency_msgs = [
        "My father collapsed and is not breathing, what do I do?",
        "I have crushing chest pain and my left arm is numb.",
        "My child swallowed a handful of pills and is drowsy.",
    ]
    # Messages that should all be classified INFO.
    info_msgs = [
        "What are the opening hours of public clinics in Kisumu?",
        "Find a notable public hospital in Nakuru, Kenya.",
    ]

    print("\n=== EMERGENCY messages (agent must NOT be called) ===")
    before = agent_call_count  # snapshot the counter before running emergencies
    for msg in emergency_msgs:
        print(f"\nMessage: {msg}")
        print(route_patient_query(msg))
    # If the counter did not move, no emergency reached the agent. Otherwise the test fails.
    assert agent_call_count == before, "FAIL: an emergency message reached the dynamic agent"
    print("\nPASS: no emergency message reached the dynamic agent.")

    print("\n=== INFO messages (agent should be called) ===")
    before = agent_call_count  # snapshot again before running INFO messages
    for msg in info_msgs:
        print(f"\nMessage: {msg}")
        print(route_patient_query(msg))
    # Each INFO message should have bumped the counter by one.
    assert agent_call_count == before + len(info_msgs), "FAIL: INFO messages did not all reach the agent"
    print("\nPASS: INFO messages were routed to the dynamic agent.")


# Only run the tests when this file is executed directly, not when imported.
if __name__ == "__main__":
    run_tests()