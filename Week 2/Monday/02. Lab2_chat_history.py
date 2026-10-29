# lab2_chat_history.py

import os                                      # standard library, used for environment access
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_openai import ChatOpenAI        # the OpenAI chat model wrapper
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage  # typed chat message objects

# Load the .env file (for example OPENAI_API_KEY) into the environment so the client finds the key.
load_dotenv()

# Create the chat model. temperature=0.2 keeps answers consistent and fairly deterministic.
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.2)

# Start the conversation as a list of messages.
#   SystemMessage sets the assistant's role for the whole chat.
#   HumanMessage is the user's first question.
chat_history = [
    SystemMessage(content="You are an AfyaPlus clinic scheduling assistant in Kisumu."),
    HumanMessage(content="Hi, do you have slots open tomorrow morning?")
]

# Send the current history to the model and get the first reply (an AIMessage).
first_reply = llm.invoke(chat_history)
print(f"AI: {first_reply.content}\n")

# Manually keep the memory going: append the model's reply, then the user's next turn.
# The model has no memory of its own, so each turn must be added back to the list
# for the next invoke call to see the full context.
chat_history.append(AIMessage(content=first_reply.content))
chat_history.append(HumanMessage(content="Great, lock in the 9:00 AM slot for me please."))

# Send the now-longer history. Because the booking request and the earlier
# context are both in the list, the model answers with that context in mind.
second_reply = llm.invoke(chat_history)
print(f"AI: {second_reply.content}")

# Optional third turn, currently commented out. Uncomment to continue the chat.
# The pattern is the same: append the last AI reply, append the new user message,
# then invoke again on the full history.
# Note: this example line contains personal details (name, patient ID, phone);
# avoid sending real patient data to the model in practice.
chat_history.append(AIMessage(content=second_reply.content))
chat_history.append(HumanMessage(content="I am John, My patient ID is +2547123456789. My number 123. I want to see a Physician."))

third_reply = llm.invoke(chat_history)
print(f"AI: {third_reply.content}")