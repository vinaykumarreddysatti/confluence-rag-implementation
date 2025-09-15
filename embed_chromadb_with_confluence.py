import chromadb
import os
import requests
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from requests.auth import HTTPBasicAuth
from dotenv import load_dotenv
from chromadb.config import Settings


class EmbedChromaDBWithConfluenceContent:
    def __init__(self):
        load_dotenv() 
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        self.chroma_client = chromadb.PersistentClient(path="./confluence_chroma_db", settings=Settings(allow_reset=True))
        try:
            self.collection = self.chroma_client.get_collection("confluence_docs")
        except:
            self._embed_and_store()
    
    def _get_datata_from_confluence(self):
        auth = HTTPBasicAuth(os.environ.get('CONFLUENCE_USERNAME'), os.environ.get('CONFLUENCE_API_TOKEN'))
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json"
        }
        url = f'{os.environ.get('CONFLUENCE_DOMAIN')}/wiki/api/v2/pages?body-format=storage'
        response = requests.get(url, headers=headers, auth=auth)
        pages = response.json()["results"]
        for page in pages:
            self.title = page["title"]
            self.content = page["body"]["storage"]["value"]
            print(self.title, self.content[:200])
    
    def _chunk_data(self):
        chunks = self.splitter.split_text(self.content)
        return chunks

    def _embed_and_store(self):
        self._get_datata_from_confluence()
        chunks = self._chunk_data()
        EmbedChromaDBWithConfluenceContent.collection = self.chroma_client.create_collection("confluence_docs")
        embeddings = self.model.encode(chunks)
        for i, chunk in enumerate(chunks):
            EmbedChromaDBWithConfluenceContent.collection.upsert(documents=[chunk], embeddings=[embeddings[i]], ids=[str(i)])
    def get_collection(self):
        return self.collection
    def delete_db(self):
        self.chroma_client.reset()