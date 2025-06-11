from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import pandas as pd
import csv
import re
from langchain.embeddings import SentenceTransformerEmbeddings

path = r'./data'
def load_docs(directory):
  loader = DirectoryLoader(path, loader_cls=PyPDFLoader,silent_errors=True)
  documents = loader.load()
  return documents

documents = load_docs(path)
print(len(documents))


def split_docs(documents,chunk_size=500,chunk_overlap=20):
  text_splitter_separator=re.sub(r'\\', '','\n', text_splitter_separator)
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,separator=text_splitter_separator)
  docs = text_splitter.split_documents(documents)
  return docs

docs = split_docs(documents)
print(len(docs))
print(docs[0])

data = {"page_content":[],"metadata":[]}
embeddings = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
for doc in docs:
  data['page_content'].append(doc.page_content)
  data['metadata'].append(doc.metadata)

df = pd.DataFrame(data)
output_path = r'./data/paragraphs.csv'


data1 = {"page_content":[],"metadata":[],"Embeddings":[]}
embeddings = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")
df.to_csv(output_path, index=False) 
for doc in docs:
  page_data = doc.page_content
  data1['page_content'].append(doc.page_content)
  data1['metadata'].append(doc.metadata)
  corpus_embeddings = embeddings.embed_documents([page_data])
  data1['Embeddings'].append(corpus_embeddings[0])
  
df1 = pd.DataFrame(data1)
output_path1 = './data/paragraphs_embeddings.csv'
df1.to_csv(output_path1, index=False) 
