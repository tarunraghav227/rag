import os
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

# load_dotenv()

def load_documents(directory_path="docs"):
    loader = DirectoryLoader(
        directory_path, 
        glob="**/*.txt", 
        show_progress=True
    )
    documents = loader.load()

    if len(documents) == 0:
        raise FileNotFoundError(f"No documents found in the directory: {directory_path}")


    for i, doc in enumerate(documents[:5]):
        print("\n")
        print("-" * 50)
        print(f"Document {i+1}:")
        print(f"Content length: {len(doc.page_content)}")
        print(f"Content preview: {doc.page_content[:100]}...")  # Print the first 100 characters of the document
        print(f"Source: {doc.metadata['source']}")
        print(f"Metadata: {doc.metadata}")
        print("-" * 50)

    return documents

def chunk_documents(documents, chunk_size=800, chunk_overlap=0):
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)

    for i, chunk in enumerate(chunks[:5]):
        print("\n")
        print("-" * 50)
        print(f"Chunk {i+1}:")
        print(f"Content length: {len(chunk.page_content)}")
        print(f"Content preview: {chunk.page_content[:100]}...")  # Print the first 100 characters of the chunk
        print(f"Source: {chunk.metadata['source']}")
        print(f"Metadata: {chunk.metadata}")
        print("-" * 50)

    if len(chunks) > 5:
        print(f"\n... and {len(chunks) - 5} more chunks.")

    return chunks

def create_vector_store(chunks, persist_directory="db/chroma_db"):
    print(f"\nCreating vector store in directory: {persist_directory}")
    embedding_model = OpenAIEmbeddings(model="text-embedding-3-small")

    print("\nGenerating embeddings for the chunks...")
    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=persist_directory,
        collection_metadata={"hnsw:space": "cosine"}
    )

    print("\nVector store created successfully.")

    return vector_store

def main():
    print("Starting the ingestion pipeline...")

    #load docs
    documents = load_documents()

    #chunking
    chunks = chunk_documents(documents)

    #embedding
    create_vector_store(chunks)

    #store in chroma

if __name__ == "__main__":
    main()
 