import os
import json
import streamlit as st
import chromadb
from chromadb.config import Settings
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables from .env (OPENAI_API_KEY)
load_dotenv()

# Initialize OpenAI client (it will read OPENAI_API_KEY from environment)
client = OpenAI()

# Initialize Chroma DB (persistent DB on disk)
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="dali_museum")

def rag_answer(question: str) -> str:
    """Retrieve-augmented answer about The Dalí Museum."""
    # 1) Create question embedding
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    # 2) Retrieve similar chunks from Chroma
    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )
    retrieved_text = " ".join(results["documents"][0])

    # 3) Build RAG prompt
    prompt = f"""
You are an assistant answering questions about The Dalí Museum.
Use ONLY the context below.

CONTEXT:
{retrieved_text}

QUESTION:
{question}

ANSWER:
"""

    # 4) Generate final answer with LLM
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


# ---------------- Streamlit UI ----------------

st.title("🎨 Dalí Museum — RAG Assistant")
st.write("Ask a question about The Dalí Museum.")

user_question = st.text_input("Your question:")

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        try:
            answer = rag_answer(user_question)
            st.success(answer)
        except Exception as e:
            st.error(f"Error while generating answer: {e}")
