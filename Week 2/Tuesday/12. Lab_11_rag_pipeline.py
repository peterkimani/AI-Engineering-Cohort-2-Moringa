# rag_pipeline.py
# End-to-end RAG pipeline reading YOUR documents from a knowledge folder.
#
# Put this file in afyaplus/, create a "knowledge" folder beside it, drop your
# .txt documents in, then run:  python rag_pipeline.py
# Requires: langchain-openai, langchain-community, langchain-chroma,
#           langchain-text-splitters, python-dotenv

import os                                      # standard library, used for file paths and checks
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from langchain_community.document_loaders import DirectoryLoader, TextLoader  # load many files from a folder
from langchain_text_splitters import RecursiveCharacterTextSplitter  # splits long text into overlapping chunks
from langchain_openai import ChatOpenAI, OpenAIEmbeddings  # chat model + the embedding model
from langchain_chroma import Chroma            # persistent on-disk vector store
from langchain_core.prompts import ChatPromptTemplate  # builds the prompt with placeholders
from langchain_core.output_parsers import StrOutputParser  # turns the model's message into a plain string
from langchain_core.runnables import RunnablePassthrough  # passes an input through a chain unchanged

# Load the .env file (for example OPENAI_API_KEY) into the environment so the clients find the key.
load_dotenv()

# You create KNOWLEDGE_DIR and fill it with documents. Chroma creates PERSIST_DIR itself.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
KNOWLEDGE_DIR = os.path.join(BASE_DIR, "knowledge")
PERSIST_DIR = os.path.join(BASE_DIR, "chroma_db")


def build_vector_store() -> Chroma:
    """Builds the Chroma vector store from every document in KNOWLEDGE_DIR.

    Steps, matching the lab's four stages:
      1. Load: DirectoryLoader reads every .txt file in the folder into Document
         objects. glob="**/*.txt" also picks up files in subfolders.
      2. Split: RecursiveCharacterTextSplitter breaks the combined text into
         chunks of about 500 characters with 50 of overlap. Overlap stops a
         sentence spanning a chunk boundary from being cut in half.
      3. Embed and store: each chunk is embedded and written into a Chroma store
         that persists to PERSIST_DIR, so later runs can reuse the vectors.

    Raises a clear error if the folder is missing or empty, so you know to
    create it and add documents rather than getting a confusing failure later.
    """
    # Fail early with a helpful message if the knowledge folder is not set up.
    if not os.path.isdir(KNOWLEDGE_DIR):
        raise FileNotFoundError(
            f"Folder '{KNOWLEDGE_DIR}' not found. Create it beside this script "
            f"and add your .txt documents."
        )

    # Step 1: load every .txt file in the folder (and subfolders).
    loader = DirectoryLoader(
        KNOWLEDGE_DIR,
        glob="**/*.txt",
        loader_cls=TextLoader,
        loader_kwargs={"encoding": "utf-8"},
    )
    documents = loader.load()
    if not documents:
        raise ValueError(
            f"No .txt documents found in '{KNOWLEDGE_DIR}'. Add some files and rerun."
        )

    # Step 2: split into right-sized, overlapping chunks.
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = splitter.split_documents(documents)

    # Step 3: embed the chunks and persist them in Chroma.
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=PERSIST_DIR,
    )
    print(f"Indexed {len(documents)} document(s) into {len(chunks)} chunks.")
    return vector_store


def build_rag_chain(vector_store: Chroma):
    """Composes the grounded answer chain from a vector store.

    Step 4: expose a retriever with a chosen k (here 3), then wire it into the
    prompt, model, and parser. k is how many chunks are fetched per query; a
    larger k gives more context but more noise and cost. The prompt forces the
    model to answer only from the retrieved context, keeping answers grounded.
    """
    # Step 4a: retriever with an explicit k.
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})

    # Step 4b: grounded prompt. The refusal instruction is the guardrail.
    prompt = ChatPromptTemplate.from_template(
        "Answer using only this context. If not found, say 'Information not found.'\n"
        "Context: {context}\nQuestion: {question}"
    )
    llm = ChatOpenAI(model="gpt-5.4-mini", temperature=0.0)

    # The chain: question -> (retriever for context, passthrough for question) -> prompt -> model -> string.
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return rag_chain


if __name__ == "__main__":
    vector_store = build_vector_store()
    rag_chain = build_rag_chain(vector_store)

    # Replace this with a question your own documents can answer.
    question = "How soon is staff travel reimbursed?"
    print(f"\nQ: {question}")
    print(rag_chain.invoke(question))