# lab3_sequential_chain.py

import os                                      # standard library, used for environment access
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_openai import ChatOpenAI        # the OpenAI chat model wrapper
from langchain_core.prompts import ChatPromptTemplate      # builds a reusable prompt with placeholders
from langchain_core.output_parsers import StrOutputParser  # turns the model's message into a plain string

# Load the .env file (for example OPENAI_API_KEY) into the environment so the client finds the key.
load_dotenv()

# Create the chat model. temperature=0.3 keeps replies focused with a little variation.
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.3)

# Define the prompt template. {topic} is a placeholder filled in at invoke time,
# so the same template can be reused for any subject.
prompt = ChatPromptTemplate.from_template(
    "Draft a one-sentence administrative acknowledgement for an AfyaPlus patient about: {topic}"
)

# The parser extracts the plain text out of the model's AIMessage response.
parser = StrOutputParser()

# Build the chain with the LCEL pipe operator. Data flows left to right:
#   prompt : fills the template with the input dict
#   llm    : sends the filled prompt to the model
#   parser : converts the model's reply into a string
chain = prompt | llm | parser   # the LCEL pipe

# Run the chain. The dict supplies the {topic} value, and the final output
# is a plain string because the parser is the last step.
result = chain.invoke({"topic": "a consistent headache"})
print(result)