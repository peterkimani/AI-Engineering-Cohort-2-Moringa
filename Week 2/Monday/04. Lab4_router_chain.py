# lab4_router_chain.py

import os                                      # standard library, used for environment access
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_openai import ChatOpenAI        # the OpenAI chat model wrapper
from langchain_core.prompts import ChatPromptTemplate      # builds a reusable prompt with placeholders
from langchain_core.output_parsers import StrOutputParser  # turns the model's message into a plain string

# Load the .env file (for example OPENAI_API_KEY) into the environment so the client finds the key.
load_dotenv()

# Create the router model.
#   temperature=0.0 removes randomness, so the same message always gets the same label.
#                   This is what makes the router a reliable, repeatable safety gate.
#   max_tokens=1    limits the reply to a single token to keep it short and cheap.
router_llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.0)

# Prompt that asks for exactly one classification word. {message} is filled at invoke time.
router_prompt = ChatPromptTemplate.from_template(
    "Classify the patient message as exactly one word, either CLINICAL or BILLING.\n"
    "Message: {message}\nClassification:"
)

# Build the router chain: fill the prompt, send it to the model, parse the reply to a string.
router_chain = router_prompt | router_llm | StrOutputParser()

# Run the chain on one message, then normalise the output:
#   .strip()  removes surrounding whitespace or newlines
#   .upper()  makes the label uppercase so comparisons like == "BILLING" are reliable
route = router_chain.invoke({"message": "I have a headache and unpaid medical bill"}).strip().upper()
print(f"Routed to: {route}")