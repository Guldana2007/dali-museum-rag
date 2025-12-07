import streamlit as st
import chromadb
from chromadb.config import Settings
from openai import OpenAI
import json

# Initialize OpenAI client
client = OpenAI()

# Load Chroma DB
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="dali_museum")

# RAG function
def rag_answer(question: str) -> str:
    # Embed question
    embedding = client.embeddings.create(
        model="text-embedding-3-small",
        input=question
    ).data[0].embedding

    # Retrieve similar chunks
    results = collection.query(
        query_embeddings=[embedding],
        n_results=3
    )

    retrieved_text = " ".join(results["documents"][0])

    # Build prompt
    prompt = f"""
You are an assistant answering questions about The Dalí Museum.
Use ONLY the context below.

CONTEXT:
{retrieved_text}

QUESTION:
{question}

ANSWER:
"""

    # Generate answer
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content.strip()


# Streamlit UI
st.title("🎨 Dalí Museum — RAG Assistant")
st.write("Ask a question about The Dalí Museum.")

user_question = st.text_input("Your question:")

if st.button("Get Answer"):
    if user_question.strip() == "":
        st.warning("Please enter a question.")
    else:
        answer = rag_answer(user_question)
        st.success(answer)
