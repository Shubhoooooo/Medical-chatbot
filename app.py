from flask import Flask, render_template, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

# Initialize Flask app
app = Flask(__name__)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
PINECONE_ENVIRONMENT = os.environ.get("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.environ.get("PINECONE_INDEX_NAME")
PINECONE_HOST = os.environ.get("PINECONE_HOST")  # Optional

# Set environment variables
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY
os.environ["PINECONE_ENVIRONMENT"] = PINECONE_ENVIRONMENT
os.environ["PINECONE_HOST"] = PINECONE_HOST

# Debug
print("✅ Pinecone Index:", PINECONE_INDEX_NAME)
print("✅ Environment:", PINECONE_ENVIRONMENT)

# Load embeddings (from Hugging Face via helper.py)
embeddings = download_hugging_face_embeddings()

# Connect to Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=PINECONE_INDEX_NAME,
    embedding=embeddings
)

# Create retriever
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-pro-latest",
    temperature=0.4
)

# Setup prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# Create RAG pipeline (Retriever + Generator)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# Routes
@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["POST"])
def chat():
    msg = request.form.get("msg", "").strip()
    print("📩 User input:", msg)

    if not msg:
        return "Please enter a message."

    try:
        response = rag_chain.invoke({"input": msg})
        answer = response.get("answer", "Sorry, I couldn't find an answer.")
        print("✅ AI Answer:", answer)

        print("📄 Retrieved Documents:")
        for i, doc in enumerate(response.get("context", [])):
            print(f"\n-- Chunk {i+1} --\n{doc.page_content[:500]}")

        return str(answer)

    except Exception as e:
        print("❌ Error occurred:", str(e))
        return "Oops! Something went wrong while processing your query."

# Run Flask App
if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
