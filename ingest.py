from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain.text_splitter import RecursiveCharacterTextSplitter
from pymongo import MongoClient

DATA_PATH = 'data/'
ATLAS_CONNECTION_STRING = "mongodb+srv://<USERNAME>:<PASSWORD>@vectordb.jmofzvf.mongodb.net/"
# Connect to your Atlas cluster
client = MongoClient(ATLAS_CONNECTION_STRING)

# Define collection and index name
db_name = "pet_care"
collection_name = "vector_db"
atlas_collection = client[db_name][collection_name]
vector_search_index = "vector_index"

# Create vector database
def create_vector_db():
    loader = DirectoryLoader(DATA_PATH,
                             glob='*.pdf',
                             loader_cls=PyPDFLoader)

    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500,
                                                   chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embed_model = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-large",
                                       model_kwargs={'device': 'cpu'})

    # Create the vector store
    vector_search = MongoDBAtlasVectorSearch.from_documents(
        documents = texts,
        embedding = embed_model,
        collection = atlas_collection,
        index_name = vector_search_index
    )

if __name__ == "__main__":
    create_vector_db()