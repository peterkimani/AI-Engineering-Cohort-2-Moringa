# challenge13_grounding.py
# Core skill (from Lab 11): a retriever feeding a grounded prompt in an LCEL chain.
# This file shows the failure first (ungrounded), then the fix (grounded), so you
# can see the difference the guardrail makes.

import os                                      # standard library, used for environment access
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # chat model + the embedding model
from langchain_core.vectorstores import InMemoryVectorStore  # stores text as vectors in memory
from langchain_core.prompts import ChatPromptTemplate  # builds the prompt with placeholders
from langchain_core.output_parsers import StrOutputParser  # turns the model's message into a plain string
from langchain_core.runnables import RunnablePassthrough  # passes an input through a chain unchanged

# Load the .env file (for example OPENAI_API_KEY) into the environment so the clients find the key.
load_dotenv()

# The store holds a single in-scope fact. A question about anything else is
# "out of scope" and a grounded chain should refuse to answer it.
store = InMemoryVectorStore(OpenAIEmbeddings())
store.add_texts(["AfyaPlus reimburses staff travel within 30 days of an approved claim."])
retriever = store.as_retriever()

llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.0)
parser = StrOutputParser()

# ---------------------------------------------------------------------------
# BEFORE: ungrounded prompt (demonstrates the failure)
# ---------------------------------------------------------------------------
# This prompt never tells the model to stay inside the context, so for an
# off-topic question the model falls back on its training and fabricates an answer.
ungrounded_prompt = ChatPromptTemplate.from_template(
    "Context: {context}\nQuestion: {question}"
)
ungrounded_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | ungrounded_prompt | llm | parser
)

# ---------------------------------------------------------------------------
# AFTER: grounded prompt (the guardrail)
# ---------------------------------------------------------------------------
# The guardrail is this instruction: answer ONLY from the context, and if the
# answer is not there, reply with the exact refusal phrase. The fix lives in the
# prompt, not the retriever.
grounded_prompt = ChatPromptTemplate.from_template(
    "Answer using ONLY the context. If the answer is not in it, reply exactly: "
    "Information not found.\nContext: {context}\nQuestion: {question}"
)
grounded_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | grounded_prompt | llm | parser
)

if __name__ == "__main__":
    out_of_scope = "Best tourist beaches in Mombasa?"
    in_scope = "How soon is staff travel reimbursed?"

    # Show the failure: no grounding instruction, so the model invents an answer.
    print("=== BEFORE guardrail (ungrounded) ===")
    print(ungrounded_chain.invoke(out_of_scope))

    # Show the fix: same question, but now the model refuses because the answer
    # is not in the context.
    print("\n=== AFTER guardrail (grounded) ===")
    print(grounded_chain.invoke(out_of_scope))   # -> Information not found.

    # Confirm the guardrail did not break in-scope answers.
    print("\n=== AFTER guardrail, in-scope question ===")
    print(grounded_chain.invoke(in_scope))        # -> within 30 days