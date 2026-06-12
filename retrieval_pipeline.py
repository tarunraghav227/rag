from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage

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


combined_document = "\n\n".join([doc.page_content for doc in results])

prompt = f"Answer the following question based on the provided documents:\n\n{combined_document}\n\nQuestion: {query} \n Please provide a concise answer. If the answer is not found in the documents, say 'Answer not found in the provided documents.'"

model = ChatOpenAI(model="gpt-4o", temperature=0.2)

messages = [
    SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents."),
    HumanMessage(content=prompt)
]


result = model.invoke(messages)

print("\nAnswer:")
print(result)