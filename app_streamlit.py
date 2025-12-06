import os
from typing import List

import streamlit as st
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

# -----------------------------
# 1. Page config
# -----------------------------
st.set_page_config(
    page_title="Dalí Museum RAG Assistant",
    page_icon="🎨",
    layout="wide",
)

st.title("🎨 Dalí Museum — RAG Assistant")
st.write(
    "Ask questions about the Dalí Museum. "
    "The assistant retrieves information from a small curated dataset and "
    "uses an OpenAI chat model to generate grounded answers."
)

# -----------------------------
# 2. OpenAI API key handling
# -----------------------------
st.sidebar.header("API configuration")

default_key = os.environ.get("OPENAI_API_KEY", "")
api_key_input = st.sidebar.text_input(
    "OpenAI API key",
    type="password",
    value=default_key,
    help="Your OpenAI API key (starts with 'sk-').",
)

if api_key_input:
    os.environ["OPENAI_API_KEY"] = api_key_input

if not os.environ.get("OPENAI_API_KEY"):
    st.warning("Please provide your OpenAI API key in the sidebar.")
    st.stop()

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# -----------------------------
# 3. Dataset (same as in notebook)
# -----------------------------
documents = [
    {
        "id": "1",
        "title": "About the Dalí Museum",
        "section": "overview",
        "text": (
            "The Dalí Museum is located in St. Petersburg, Florida. "
            "It is dedicated to the works of the Spanish surrealist artist Salvador Dalí. "
            "The museum holds one of the largest collections of Dalí’s works outside Europe."
        ),
    },
    {
        "id": "2",
        "title": "Visitor Information",
        "section": "visit",
        "text": (
            "The museum offers guided tours, audio guides, educational programs, "
            "a garden inspired by Dalí’s work, and a museum shop with books and souvenirs."
        ),
    },
    {
        "id": "3",
        "title": "Tickets and Hours",
        "section": "tickets",
        "text": (
            "The Dalí Museum is open daily from 10 AM to 6 PM. "
            "Tickets can be purchased online or at the entrance. "
            "Discounts are available for students and seniors."
        ),
    },
    {
        "id": "4",
        "title": "Exhibitions",
        "section": "exhibitions",
        "text": (
                "The museum hosts rotating exhibitions showcasing Dalí’s paintings, drawings, "
                "sculptures, and interactive displays."
        ),
    },
    {
        "id": "5",
        "title": "Location",
        "section": "location",
        "text": (
            "The museum is located at 1 Dalí Boulevard, St. Petersburg, Florida, "
            "near the city waterfront."
        ),
    },
]

# -----------------------------
# 4. ChromaDB setup (in-memory)
# -----------------------------
chroma_client = chromadb.Client()

embedding_fn = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.environ["OPENAI_API_KEY"],
    model_name="text-embedding-3-small",
)

# recreate collection on each app run
try:
    chroma_client.delete_collection("dali_museum")
except Exception:
    pass

collection = chroma_client.create_collection(
    name="dali_museum",
    embedding_function=embedding_fn,
)

collection.add(
    ids=[doc["id"] for doc in documents],
    documents=[doc["text"] for doc in documents],
    metadatas=[{"title": doc["title"], "section": doc["section"]} for doc in documents],
)

# -----------------------------
# 5. RAG helpers
# -----------------------------
def rag_search(query: str, k: int = 3):
    """Retrieve top-k documents from ChromaDB for a given query."""
    results = collection.query(query_texts=[query], n_results=k)
    return results


def llm_answer(question: str, context: str) -> str:
    """Generate an answer using OpenAI chat model with the provided context."""
    prompt = f"""
You are a helpful museum assistant specializing in Salvador Dalí and The Dalí Museum.
Use ONLY the information from the CONTEXT to answer the QUESTION.
If the answer is not in the context, say exactly:
"The context does not contain this information."

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
    )

    return response.choices[0].message.content.strip()


def rag_answer(question: str, k: int = 3):
    """Run the full RAG pipeline: retrieve context and generate an answer."""
    search_results = rag_search(question, k=k)
    docs = search_results["documents"][0]
    metas = search_results["metadatas"][0]

    context_parts = []
    for meta, doc in zip(metas, docs):
        title = meta.get("title", "")
        section = meta.get("section", "")
        header = f"[{title} | {section}]" if title or section else ""
        context_parts.append(f"{header}\n{doc}")

    context = "\n\n".join(context_parts)
    answer = llm_answer(question, context)

    return answer, context, metas, docs

# -----------------------------
# 6. Streamlit UI
# -----------------------------
st.sidebar.header("RAG configuration")
top_k = st.sidebar.slider("Number of retrieved chunks (k)", min_value=1, max_value=5, value=3)

user_question = st.text_input(
    "Ask a question about the Dalí Museum:",
    placeholder="For example: Where is the Dalí Museum located?",
)

if st.button("Get answer"):

    if not user_question.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Retrieving context and generating answer..."):
            answer, context, metas, docs = rag_answer(user_question, k=top_k)

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🧠 Answer")
            st.write(answer)

        with col2:
            st.subheader("📚 Retrieved context")
            for meta, doc in zip(metas, docs):
                st.markdown(f"**{meta.get('title', '')}** — _{meta.get('section', '')}_")
                st.write(doc)
                st.markdown("---")
else:
    st.info("Enter a question and click **Get answer** to start.")
