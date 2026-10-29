# challenge12_procurement_rag.py
# Core skill (from Lab 11): extend a vector index and retrieve the right chunk.

import os                                      # standard library, used for environment access
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # chat model + the embedding model
from langchain_core.vectorstores import InMemoryVectorStore  # stores text as vectors in memory
from langchain_core.prompts import ChatPromptTemplate  # builds the prompt with placeholders
from langchain_core.output_parsers import StrOutputParser  # turns the model's message into a plain string
from langchain_core.runnables import RunnablePassthrough  # passes an input through a chain unchanged

# Load the .env file (for example OPENAI_API_KEY) into the environment so the clients find the key.
load_dotenv()

# ---------------------------------------------------------------------------
# Step 1: process the procurement policy into a list of text chunks
# ---------------------------------------------------------------------------
# Each string is one retrievable chunk. Splitting the policy by rule (rather
# than storing one long block) lets the retriever return just the relevant rule
# for a question instead of the whole document.
procurement_policy = [
    "Procurement under $500 can be approved by the team lead alone.",
    "Logistics hardware purchases between $1,000 and $2,000 require the Procurement Officer's signature.",
    "Any purchase above $2,000 requires both the Procurement Officer and the Finance Director to sign off.",
]

# ---------------------------------------------------------------------------
# Step 2: initialise embeddings and push the texts into the vector store
# ---------------------------------------------------------------------------
embeddings = OpenAIEmbeddings()               # turns each chunk into a meaning vector
vector_store = InMemoryVectorStore(embeddings)  # holds the vectors and searches them by similarity
# add_texts embeds every chunk and stores it. This is the "extend the index"
# step: calling it again later would add more policy without rebuilding the store.
vector_store.add_texts(procurement_policy)
retriever = vector_store.as_retriever()        # embeds a question and returns the closest chunks

# ---------------------------------------------------------------------------
# Step 3: construct the LCEL rag_chain (retriever + prompt + model)
# ---------------------------------------------------------------------------
# The prompt forces the model to answer only from the retrieved context, so it
# reports the policy rather than guessing.
prompt = ChatPromptTemplate.from_template(
    "Answer using only this context. If not found, say 'Information not found.'\n"
    "Context: {context}\nQuestion: {question}"
)
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.0)

# The chain: the question is used two ways at once. It goes to the retriever to
# become {context}, and RunnablePassthrough forwards it unchanged to {question}.
# Then the prompt is filled, the model answers, and the parser returns a string.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# ---------------------------------------------------------------------------
# Step 4: invoke and test with a question answerable only from the new policy
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    query = "What approval signature is needed for a logistics hardware purchase of $1,500?"

    # Confirm the retriever pulled the right chunk (the $1,000-$2,000 rule).
    print("--- Retrieved context ---")
    for doc in retriever.invoke(query):
        print("-", doc.page_content)

    # Run the full chain and print the grounded answer.
    response = rag_chain.invoke(query)
    print(f"\nFinal Grounded Generation Response:\n{response}")