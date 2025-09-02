# RAG-PDF-Reader

A Streamlit application that can read PDF files and generate output for the questions asked by using LLM.

## Prerequisites

1. Pyhton 3.12 or higher
2. OpenAI API Key

## Setup

1. Clone the repository
```
git clone [repository-url]
cd rag-pdf-reader
```

2. Create `.env` file and enter the OpenAI API Key
```
OPENAI_API_KEY=your_openai_api_key
```

## Execution

1. Create docker image
```
$ docker build -t rag-pdf-reader .
```

2. Run the application
```
$ docker run -p 8501:8501 rag-pdf-reader
```
