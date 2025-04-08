'''from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os


load_dotenv()

PINECONE_API_KEY=os.environ.get('PINECONE_API_KEY')
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY


extracted_data=load_pdf_file(data='Data/')
text_chunks=text_split(extracted_data)
embeddings = download_hugging_face_embeddings()


pc = Pinecone(api_key=PINECONE_API_KEY)

index_name = "medicalbot"


pc.create_index(
    name=index_name,
    dimension=384, 
    metric="cosine", 
    spec=ServerlessSpec(
        cloud="aws", 
        region="us-east-1"
    ) 
) 

# Embed each chunk and upsert the embeddings into your Pinecone index.
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings, 
)'''
from src.helper import load_pdf_file, text_split, download_hugging_face_embeddings
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()

# Get Pinecone API key
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise EnvironmentError("PINECONE_API_KEY not set in environment variables.")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Step 1: Load and process PDFs
print("Loading PDF data from 'Data/'...")
extracted_data = load_pdf_file(data="Data/")

if not extracted_data.strip():
    raise ValueError("No text extracted from PDF. Please check the PDF content.")

# Step 2: Split into chunks
print("Splitting text into chunks...")
text_chunks = text_split(extracted_data)
print(f"Total Chunks Created: {len(text_chunks)}")

# Step 3: Load embedding model
print("Downloading embedding model...")
embeddings = download_hugging_face_embeddings()

# Step 4: Initialize Pinecone client
print("Connecting to Pinecone...")
pc = Pinecone(api_key=PINECONE_API_KEY)

# Step 5: Create index if not exists
index_name = "medicalbot"
if index_name not in pc.list_indexes().names():
    print(f"Creating Pinecone index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,  # All-MiniLM-L6-v2 uses 384-dimensional vectors
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
else:
    print(f"Index '{index_name}' already exists. Skipping creation.")

# Step 6: Store documents in Pinecone
print("Uploading chunks to Pinecone...")
docsearch = PineconeVectorStore.from_documents(
    documents=text_chunks,
    index_name=index_name,
    embedding=embeddings,
)

print("All data successfully embedded and upserted into Pinecone.")
