import os
import tempfile
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Qdrant
from langchain.chat_models import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

QDRANT_PATH = "./local_qdrant"
COLLECTION_NAME = "my_collection"

def get_apikey():
    load_dotenv()
    api_key = os.environ.get("OPENAI_API_KEY")
    return api_key

def get_text_chunks(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return text_splitter.split_documents(documents)
  
def upload_and_get_chunks():
  files = st.file_uploader("Upload files in PDF format.", type="pdf", key="my_file_uploader",accept_multiple_files=True)
  chunks = []
  if not files:
    return None
  else:
    for input_file in files:
      with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(input_file.getvalue())
        tmp_file_path = tmp_file.name
        loader = PyPDFLoader(file_path=tmp_file_path)
        documents = loader.load()
        tmp_chunks = get_text_chunks(documents)
        chunks.extend(tmp_chunks)
    os.unlink(tmp_file.name)    
    return chunks

def get_vector_store(texts):
    qdrant_client = QdrantClient(path=QDRANT_PATH)

    collections = qdrant_client.get_collections().collections

    collection_names = [collection.name for collection in collections]
    if COLLECTION_NAME not in collection_names:
        qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

    texts = [doc.page_content for doc in texts if hasattr(doc, 'page_content')]

    vector_store = Qdrant(
        client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=OpenAIEmbeddings()
    )
    vector_store.add_texts(texts)
    return vector_store

def initialize_rag_chain(model_name, vector_store):
    llm = ChatOpenAI(model_name=model_name)
    if "memory" not in st.session_state:
        st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    rag_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory = st.session_state.memory
    )
    return rag_chain

def process_user_question(rag_chain):
    user_question = st.text_input("Enter the question:", key="my_input")
    if user_question:
        response = rag_chain({"question": user_question})
        st.write("Output:", response["answer"])

def main():
    st.title("PDF Document Reader")

    api_key = get_apikey()
    if not api_key:
        st.error("Please set the OPENAI key.")
        return

    texts = upload_and_get_chunks()
    if not texts:
      return

    vector_store = get_vector_store(texts)

    rag_chain = initialize_rag_chain("gpt-4o-mini", vector_store)

    if st.button("Clear Output"):
        st.session_state.my_input = ""
        st.session_state.pop("memory")
        st.success("Output has been cleared.")

    st.title("Enter the question")
    process_user_question(rag_chain)

if __name__ == "__main__":
    main()
