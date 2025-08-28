import os
from dotenv import load_dotenv
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.document_loaders import TextLoader

# Load environment variables from .env
load_dotenv()

# Load documents
loader = TextLoader("sample_docs.txt")
documents = loader.load()

# Split documents into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Create embeddings and vector store
embeddings = OpenAIEmbeddings()
db = Chroma.from_documents(docs, embeddings)

# Set up retriever
retriever = db.as_retriever()

# Set up LLM
llm = OpenAI(temperature=0)

# Create RAG pipeline
qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# Demo query
query = "What is LangChain?"
result = qa.run(query)
print("Answer:", result)
