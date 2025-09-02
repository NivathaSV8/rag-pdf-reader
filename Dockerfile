FROM python:3.12

RUN mkdir /rag-pdf-reader
WORKDIR /rag-pdf-reader

COPY . /rag-pdf-reader

RUN pip3 install -r requirements.txt

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "data_extraction_llm.py", "--server.port=8501", "--server.address=0.0.0.0"]