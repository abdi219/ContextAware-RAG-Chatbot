# Context Aware RAG Chatbot

Developed as part of the DeveloperHub Corporation AI/ML Engineering Internship, this project builds a context aware chatbot using Retrieval Augmented Generation, LangChain, vector search, and Streamlit.

## Overview

This project implements a conversational AI system that retrieves relevant information from a custom knowledge base and generates accurate responses using a Retrieval Augmented Generation pipeline. The chatbot combines document embeddings, vector storage, language models, and conversation history to provide meaningful context aware interactions.

## Visualization

<img width="1901" height="876" alt="Screenshot 2026-07-13 205443" src="https://github.com/user-attachments/assets/51160190-ebd8-4f41-ba88-c1ba88953809" />

## Features

- Custom knowledge base creation and processing
- Document chunking for efficient retrieval
- Semantic search using Hugging Face embeddings
- Chroma vector database integration
- LangChain based RAG pipeline
- Conversational memory for maintaining context
- Streamlit based chatbot interface
- Hugging Face language model integration

## Technologies Used

- Python
- LangChain
- ChromaDB
- Hugging Face Transformers
- Sentence Transformers
- Streamlit
- Retrieval Augmented Generation

## Project Structure

```
Context-Aware-RAG-Chatbot/
│
├── Context_Aware_RAG_Chatbot.ipynb
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Working Process

1. A custom knowledge base is prepared and divided into smaller document chunks.
2. Document chunks are converted into embeddings using a sentence transformer model.
3. The embeddings are stored in a Chroma vector database.
4. User queries are matched with relevant information through semantic retrieval.
5. Retrieved context is passed to a language model to generate responses.
6. Conversation history enables context aware chatbot interactions.

## Running the Application

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the Streamlit application:

```bash
streamlit run app.py
```

## Results

The chatbot successfully retrieves relevant information from the knowledge base and generates context aware responses while maintaining conversational history.

## Future Improvements

- Support for multiple document sources
- Enhanced memory management
- Cloud based deployment
- Integration with larger language models
```
