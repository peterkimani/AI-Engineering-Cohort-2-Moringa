# lab2_4_1_ingestion.py
# RAG ingestion with LlamaIndex (a different library from LangChain, same idea):
# load documents -> split into nodes -> build a vector index -> query it.

import os                                      # standard library, used for environment and paths
from dotenv import load_dotenv                 # loads variables from a .env file into the environment
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex  # document loader + vector index
from llama_index.core.node_parser import SentenceSplitter  # splits documents into sentence-aware chunks
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings

# Load the .env file (for example OPENAI_API_KEY) into the environment.
load_dotenv()

# LlamaIndex defaults to gpt-3.5-turbo, which OpenAI has deprecated. Set a
# current model so the query step works. Settings applies it to the whole script.
Settings.llm = OpenAI(model="gpt-5.4-mini", temperature=0.0)

# Fail fast if the API key is missing. LlamaIndex uses OpenAI for both embeddings
# and the answer model by default, so without a key nothing below would work.
# Catching it here gives a clear message instead of a confusing failure later.
if not os.environ.get("OPENAI_API_KEY"):
   raise ValueError("CRITICAL: OPENAI_API_KEY is unassigned.")

# Anchor the data path to THIS script's folder, not the current working directory.
# os.path.abspath(__file__) is the full path to this file; os.path.dirname strips
# the filename to leave its folder; os.path.join adds "data" onto it. So the
# "data" folder is always found beside this script, wherever you run it from.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")

# ---------------------------------------------------------------------------
# Step 1: load the documents
# ---------------------------------------------------------------------------
print("--- Step 1: Loading raw assets via LlamaIndex ---")
# SimpleDirectoryReader reads every file in the data folder. It auto-detects
# common formats (txt, md, pdf, docx, and more), so you just point it at the folder.
reader = SimpleDirectoryReader(DATA_DIR)
documents = reader.load_data()  # returns a list of Document objects, one per file

# ---------------------------------------------------------------------------
# Step 2: split the documents into nodes (chunks)
# ---------------------------------------------------------------------------
print("--- Step 2: Segmenting text data using SentenceSplitter ---")
# SentenceSplitter cuts the text into chunks on sentence boundaries, so a chunk
# does not end mid-sentence. chunk_size=256 is the target chunk length (in tokens)
# and chunk_overlap=30 repeats ~30 tokens between neighbouring chunks so a point
# spanning a boundary is not lost. In LlamaIndex these chunks are called "nodes".
parser = SentenceSplitter(chunk_size=256, chunk_overlap=30)
nodes = parser.get_nodes_from_documents(documents)  # turn documents into a list of nodes
print(f"Successfully generated {len(nodes)} structural document nodes.")

# ---------------------------------------------------------------------------
# Step 3: build the vector index and a query engine
# ---------------------------------------------------------------------------
print("--- Step 3: Compiling VectorStore Index & Query Engine ---")
# VectorStoreIndex embeds every node and stores the vectors (in memory by default,
# so it is rebuilt each run). This one line does the embedding and storing.
index = VectorStoreIndex(nodes)

# as_query_engine wraps the index into something you can ask questions of. It
# bundles retrieval AND answering together. similarity_top_k=2 means each query
# retrieves the 2 most similar nodes (raised to 2 for Challenge 18 so a
# multi-clause question can draw on two chunks at once).
base_query_engine = index.as_query_engine(similarity_top_k=2)

# ---------------------------------------------------------------------------
# Run a test query (only when this file is executed directly)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
   # This question has two parts (privacy rules AND audit timing), which is why
   # top_k=2 helps: the engine can pull one node for each part.
   query = "What are the data privacy rules and when do administrators run security audits?"
   print(f"\nExecuting Index Query: {query}")
   # .query() does the whole RAG flow in one call: embed the question, retrieve
   # the top 2 nodes, feed them to the model, and return the grounded answer.
   response = base_query_engine.query(query)
   print(f"LlamaIndex Output:\n{response}")