# 🎨 Dalí Museum — RAG Assistant

This repository contains an end-to-end Retrieval-Augmented Generation (RAG) system built for educational purposes as part of the Module 3 assignment.

The assistant answers questions about The Dalí Museum using:
- A custom curated dataset  
- OpenAI embeddings (`text-embedding-3-small`)  
- ChromaDB as a vector store  
- OpenAI chat model (`gpt-4o-mini`)  
- Optional Streamlit application  

---

## 1. Project Structure

dali-museum-rag/
│
├── notebook/
│ └── dali_rag.ipynb # Main RAG pipeline
│
├── data/
│ └── dali_chunks.json # Custom dataset
│
├── app_streamlit.py # Optional Streamlit UI
├── requirements.txt # Dependencies
├── project.md # Assignment documentation
└── README.md # Repository overview

---

## 2. How to Run

### Notebook Version

pip install -r requirements.txt
jupyter notebook notebook/dali_rag.ipynb
or open in Google Colab:

### Streamlit UI

pip install -r requirements.txt
streamlit run app_streamlit.py
The app will open at:
http://localhost:8501

---

## 3. Features
- Custom dataset for museum knowledge  
- Dense embeddings with OpenAI  
- Fast semantic search with ChromaDB  
- Full retrieval + generation workflow  
- Clean pipeline and simple architecture  
- Optional interactive web UI via Streamlit  

---

## 4. Technologies Used
- Python  
- Jupyter Notebook / Google Colab  
- OpenAI API  
- ChromaDB  
- Streamlit  

---

## 5. Author
**Guldana Kassym-Ashim**  
AI & RPA Team Lead  

---

## 6. License
This project is intended for educational use as part of the GenAI course (Module 3).




