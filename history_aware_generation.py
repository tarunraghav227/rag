from langchain_chroma import Chroma
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

load_dotenv()

persist_directory = "db/chroma_db"
embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

db = Chroma(
    persist_directory=persist_directory, 
    embedding_function=embedding_model, 
    collection_metadata={"hnsw:space": "cosine"}
)

model = ChatOpenAI(model="gpt-4o")

chat_history = []

def ask_question(query):
    if chat_history:
        messages = [
                        "Given the previous conversation, rewrite the following question to be more specific and contextually relevant."
                    ] + chat_history + [f"New Question: {query}"]
        result = model.invoke(messages)
        search_query = result.content.strip()
        print(f"Rewritten query based on chat history: {search_query}")
    else:
        search_query = query

    retriever = db.as_retriever(search_kwargs={"k": 5})
    docs = retriever.invoke(search_query)
    
    print(f"Retrieved {len(docs)} documents for the query: '{search_query}'")
    
    for i, doc in enumerate(docs):
        print(f"\nDocument {i+1}:")
        print(f"Content length: {len(doc.page_content)}")
        print(f"Content preview: {doc.page_content[:100]}...")  # Print the first 100 characters of the document
        print(f"Source: {doc.metadata['source']}")
        print(f"Metadata: {doc.metadata}")

    combined_document = "Based on the retrieved documents, provide a concise answer to the following question:\n\n" + "\n\n".join([doc.page_content for doc in docs]) + f"\n\nQuestion: {query}. \n If the answer is not found in the documents, say 'Answer not found in the provided documents.'"

    messages =[
        SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents."),
        HumanMessage(content=combined_document)
    ]

    result = model.invoke(messages)
    chat_history.append(HumanMessage(content=query))
    chat_history.append(AIMessage(content=result.content))
    print(f"\nAnswer: {result.content}")
    return result.content

def start_chat():
    print("Welcome to the History-Aware Chatbot! Type 'exit' to quit.")
    while True:
        query = input("\nYou: ")
        if query.lower() == 'exit':
            print("Exiting the chat. Goodbye!")
            break
        answer = ask_question(query)
        print(f"Bot: {answer}")

if __name__ == "__main__":
    start_chat()

