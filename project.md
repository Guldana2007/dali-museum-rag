# Development of RAG-based AI System — Guldana Kassym-Ashim

## 1. Project Idea
This project implements a Retrieval-Augmented Generation (RAG) system focused on The Dalí Museum.  
The assistant answers user questions by retrieving relevant information from a small curated dataset and generating responses using OpenAI models.

## 2. Dataset
A custom dataset was created manually based on publicly available descriptions of:
- The museum overview  
- Visitor information  
- Tickets and hours  
- Exhibitions  
- Location  

The dataset is stored as `dali_chunks.json` and is used to build the vector database.

## 3. Architecture
- OpenAI Embedding model: `text-embedding-3-small`  
- Vector database: ChromaDB  
- LLM: `gpt-4o-mini`  
- Jupyter Notebook (Google Colab) as the development environment  
- Optional Streamlit UI for interactive demonstration  

## 4. Implementation Steps
All required steps of the Module 3 assignment are implemented:
1. Dataset preparation  
2. Embeddings generation  
3. Vector DB creation  
4. Retrieval pipeline  
5. LLM-based answer generation  
6. Full RAG assembly  
7. Evaluation with test queries  
8. Streamlit user interface  
9. Documentation and repository setup  

## 5. GitHub Repository
https://github.com/Guldana2007/dali-museum-rag

## 6. Demo Video
A short demonstration video (1–3 minutes) will be added after the Streamlit UI.

## 7. How to Run

### Notebook version:
