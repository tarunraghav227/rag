from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()

persist_directory = "db/chroma_db"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embedding_model, 
    collection_metadata={"hnsw:space": "cosine"}
)

query = "Which island spacex launched their first rocket from?"

retriever = db.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.3})

results = retriever.invoke(query)

print("\nTop 3 retrieved documents:")
for i, doc in enumerate(results):
    print("\n")
    print("-" * 50)
    print(f"Document {i+1}:")
    print(f"Content length: {len(doc.page_content)}")
    print(f"Content preview: {doc.page_content[:100]}...")  # Print the first 100 characters of the document
    print(f"Source: {doc.metadata['source']}")
    print(f"Metadata: {doc.metadata}")
    print("-" * 50)

    