# lab1_message_structures.py

import os                                      # standard library, used for environment access
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_openai import ChatOpenAI        # the OpenAI chat model wrapper
from langchain_core.messages import SystemMessage, HumanMessage  # typed message objects for a chat

# Read the .env file (for example OPENAI_API_KEY) and load it into os.environ
# so the ChatOpenAI client can pick the key up automatically.
load_dotenv()

# Create the chat model.
#   model      : which OpenAI model to call
#   temperature: 0.3 keeps replies fairly focused with a little variation
#   max_tokens : caps the reply length at 300 tokens
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.3, max_tokens=30)

# Build the conversation as an ordered list of messages.
messages = [
    # SystemMessage sets the model's role and behaviour for the whole chat.
    SystemMessage(content=(
        "You are a Senior AI Engineer specialising in medical triage interfaces "
        "at AfyaPlus Health in Kenya. Provide professional, safe, structured guidance."
    )),
    # HumanMessage is the user's actual question to the model.
    HumanMessage(content="Explain the structural difference between a deterministic chain and a dynamic agent.")
]

# Send the message list to the model and get a single AIMessage back.
response = llm.invoke(messages)

# response.content holds the model's reply text; print it to the console.
print(response.content)