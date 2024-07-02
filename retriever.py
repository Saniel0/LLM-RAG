import os.path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma

from sentence_transformers import SentenceTransformer
from typing import List

from graph_state import GraphState

# splitter function
textSplitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(chunk_size=500, chunk_overlap=0)

# Directory to persist the database
persist_directory = "./rag_database"

# load embedding model as sentence tranformer (model will be automatically downloaded from hugging face)
class MyEmbeddings:
    def __init__(self):
        self.model = SentenceTransformer("intfloat/multilingual-e5-large-instruct")

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.model.encode(t).tolist() for t in texts]
    
    # this function is needed for Chroma embeddings
    def embed_query(self, query: str) -> List[float]:
        return self.model.encode(query).tolist()

print("Loading embedding model...", end=" ", flush=True)
embeddings = MyEmbeddings()
print("DONE")

# If database does not exists, create it
if not os.path.exists(persist_directory):
    print("Importing data into vector database, this may take a while...")
    loader = TextLoader("example.txt")
    splitted = textSplitter.split_documents(loader.load())
    print(splitted)
    vectorStore = Chroma.from_documents(
        documents=splitted,
        collection_name="rag-chroma",
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Import complete, loaded ", len(splitted), " chunks")
# If it does not exist, create it and embed context
else:
    vectorStore = Chroma(
        collection_name="rag-chroma",
        embedding_function=embeddings,
        persist_directory=persist_directory,
    )

# create retriver for the database that returns 5 closest matches
retriever = vectorStore.as_retriever(search_kwargs={"k": 10})

def retrieve(question : str):
    return retriever.invoke(question)

# function for langgraph
def retrieveNODE(state : GraphState):
    print("Retrieving documents...", end=" ", flush=True)
    documents = retriever.invoke(state["question"])
    print("DONE")
    return {"documents": documents}


# for testing purposes
if __name__ == "__main__":
    print(retrieve("Where is Willowbrook?"))
