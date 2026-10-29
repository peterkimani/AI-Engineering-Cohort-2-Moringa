# lab11_rag.py
# RAG (Retrieval-Augmented Generation): the model answers from retrieved
# documents, not just its own training, so it can use your private facts.

import os                                      # standard library, used for environment access
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # chat model + the embedding model
from langchain_core.vectorstores import InMemoryVectorStore  # stores text as vectors in memory
from langchain_core.prompts import ChatPromptTemplate  # builds the prompt with placeholders
from langchain_core.output_parsers import StrOutputParser  # turns the model's message into a plain string
from langchain_core.runnables import RunnablePassthrough  # passes an input through a chain unchanged

# Load the .env file (for example OPENAI_API_KEY) into the environment so the clients find the key.
load_dotenv()

# The embedding model turns text into vectors (lists of numbers) that capture
# meaning, so that similar text ends up with similar vectors. Both the stored
# documents and the question are embedded with this.
embeddings = OpenAIEmbeddings()

# A vector store holds those vectors and can find the ones closest to a query.
# InMemoryVectorStore keeps everything in RAM, so it is wiped when the script ends.
vector_store = InMemoryVectorStore(embeddings)

# Add the knowledge base. Each string is embedded and stored. These two facts
# are the only things the model will be allowed to answer from.
vector_store.add_texts([
    "AfyaPlus reimburses staff travel within 30 days of an approved claim.",
    "Clinic equipment purchases above $500 require director sign-off."
])

# A retriever wraps the vector store with a search interface. Given a question,
# it embeds it and returns the stored texts whose vectors are most similar.
retriever = vector_store.as_retriever()

# The prompt. {context} is filled with the retrieved documents and {question}
# with the user's question. The instruction to answer ONLY from the context,
# and to say "Information not found." otherwise, is what keeps the model from
# inventing an answer (a guardrail against hallucination).
prompt = ChatPromptTemplate.from_template(
    "Answer using only this context. If not found, say 'Information not found.'\n"
    "Context: {context}\nQuestion: {question}"
)

# The chat model that writes the final answer. temperature=0.0 keeps it factual.
llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.0)

# The RAG chain, read left to right:
#   The dict builds the prompt's two inputs IN PARALLEL from the question:
#     context  : the question is sent to the retriever, which returns the
#                matching documents to fill {context}.
#     question : RunnablePassthrough forwards the original question unchanged
#                to fill {question}.
#   | prompt  : fills the template with those two values.
#   | llm     : sends the filled prompt to the model.
#   | parser  : converts the model's reply into a plain string.
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt | llm | StrOutputParser()
)

# Ask a question. The retriever finds the travel-reimbursement fact, the prompt
# puts it in {context}, and the model answers from it (expected: within 30 days).
print(rag_chain.invoke("How soon is staff travel reimbursed?"))