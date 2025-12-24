📄 Chat with PDF App

Chat with PDF is a Streamlit-based web application that allows users to upload PDF documents and ask questions about their content using AI. The app uses a Retrieval-Augmented Generation (RAG) approach to provide accurate answers strictly based on the uploaded documents.

🚀 Features

Upload one or multiple PDF files

Automatically extracts and chunks PDF text

Creates embeddings using HuggingFace MiniLM

Stores vectors locally using FAISS

Answers user questions using Groq LLM (LLaMA 3.1)

Responds only from PDF context (no hallucinations)

🛠️ Tech Stack

Frontend: Streamlit

LLM: Groq (LLaMA 3.1)

Embeddings: sentence-transformers/all-MiniLM-L6-v2

Vector Store: FAISS

Framework: LangChain (LCEL pipeline)

📌 How It Works

Upload PDF files

Click Submit & Process to build the vector index

Ask questions related to the PDF content

Get accurate answers sourced from the documents

⚠️ Note

If an answer is not found in the uploaded PDFs, the app will clearly respond:

"answer is not available in the context"

👤 Author

Pankaj Mahure
📅 © 2025
