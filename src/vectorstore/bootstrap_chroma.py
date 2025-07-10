import chromadb
from chromadb.utils import embedding_functions
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.getenv("CHROMA_API_KEY")
tenant = os.getenv("CHROMA_TENANT")
database = os.getenv("CHROMA_DATABASE")

client = chromadb.CloudClient(
    api_key=api_key,
    tenant=tenant,
    database=database
)

collection = client.get_or_create_collection(name="test_collection")

openai_ef = embedding_functions.DefaultEmbeddingFunction()

collection.add(
    documents=["ChromaDB Cloud is now initialized."],
    metadatas=[{"source": "bootstrap_script"}],
    ids=["init-001"]
)

print("Document uploaded. ChromaDB Cloud setup is now complete.")
