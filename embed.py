import os
from dotenv import load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import ChromaVectorStore
from langchain.document_loaders import TextLoader
from llama_index import GPTVectorStoreIndex
from llama_index.storage.storage_context import StorageContext
import chromadb

load_dotenv()

def embed_codebase():
    # Configure these to fit your needs
    exclude_dir = ['.git', 'node_modules', 'public', 'assets', 'Lib', 'site-packages', 'Scripts', 'env']
    exclude_files = ['package-lock.json', 'package.json', '.DS_Store']
    exclude_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ico', '.svg', '.webp',
        '.mp3', '.wav', '.env', '.json', '.md']

    # Get the local path of the repository
    local_path = os.environ.get("LOCAL_PATH")

    # Get the URL of the repository
    repo_url = os.environ.get("REPO_URL")

    # Check if there are any new files to embed
    new_files = []
    for dirpath, dirnames, filenames in os.walk(local_path):
        # Skip directories in exclude_dir
        dirnames[:] = [d for d in dirnames if d not in exclude_dir]

        for file in filenames:
            _, file_extension = os.path.splitext(file)

            # Skip files in exclude_files or with exclude_extensions
            if file not in exclude_files and file_extension not in exclude_extensions:
                new_files.append(os.path.join(dirpath, file))

    if not new_files:
        print("No new files found to embed.")
        return

    # Load and split the documents
    documents = []
    for file_path in new_files:
        # Create a relative file path
        relative_file_path = os.path.relpath(file_path, local_path)

        # Combine the repo URL and the relative file path to create a unique key
        unique_file_key = f"{repo_url}/{relative_file_path}"

        loader = TextLoader(unique_file_key, encoding='ISO-8859-1')
        documents.extend(loader.load())

    text_splitter = CharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    docs = text_splitter.split_documents(documents)

    for doc in docs:
        source = doc.metadata['source']
        cleaned_source = '/'.join(source.split('/')[1:])
        doc.page_content = "FILE NAME: " + cleaned_source + "\n###\n" + doc.page_content.replace('\u0000', '')

    # Embed the documents and store the vectors
    embeddings = OpenAIEmbeddings()

    batch_size = 50
    total_docs = len(docs)
    num_batches = total_docs // batch_size + int(total_docs % batch_size > 0)

    # Create a Chroma index
    chroma_client = chromadb.Client()
    chroma_collection = chroma_client.create_collection("quickstart")

    # Initialize ChromaVectorStore and StorageContext
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # Create the GPTVectorStoreIndex
    index = GPTVectorStoreIndex.from_documents(documents, storage_context=storage_context)

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, total_docs)
        print(f"Embedding documents {start_index + 1} to {end_index} of {total_docs}")

        batch_docs = docs[start_index:end_index]
        vectors = embeddings.embed_documents(batch_docs)

        for doc, vector in zip(batch_docs, vectors):
            vector_store.store_vector(doc.metadata['source'], vector)

        print(f"Stored batch {i + 1}/{num_batches}")

    print("Embedding complete!")