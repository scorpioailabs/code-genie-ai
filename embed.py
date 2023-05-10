import os
from dotenv import load_dotenv
from supabase.client import Client, create_client
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import SupabaseVectorStore
from langchain.document_loaders import TextLoader

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)
def embed_codebase():
    # Configure these to fit your needs
    exclude_dir = ['.git', 'node_modules', 'public', 'assets', 'Lib', 'site-packages', 'Scripts', 'env']
    exclude_files = ['package-lock.json', 'package.json', '.DS_Store']
    exclude_extensions = ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.tiff', '.ico', '.svg', '.webp',
        '.mp3', '.wav', '.env', '.json', '.md']

    # Get list of already embedded files
    try:
        response = supabase.table(os.environ.get("TABLE_NAME")).select('metadata').execute()
        already_embedded_files = [doc['metadata']['source'] for doc in response.data]
    except Exception as e:
        # Log the error
        print(e)
        already_embedded_files = []

    # Check if there are any new files to embed
    new_files = []
    for dirpath, dirnames, filenames in os.walk('repo'):
        # Skip directories in exclude_dir
        dirnames[:] = [d for d in dirnames if d not in exclude_dir]

        for file in filenames:
            _, file_extension = os.path.splitext(file)

            # Skip files in exclude_files or already embedded files
            if file not in exclude_files and file_extension not in exclude_extensions and os.path.join(dirpath, file) not in already_embedded_files:
                new_files.append(os.path.join(dirpath, file))

    if not new_files:
        print("No new files found to embed.")
        return

    # Load and split the documents
    documents = []
    for file_path in new_files:
        loader = TextLoader(file_path, encoding='ISO-8859-1')
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

    for i in range(num_batches):
        start_index = i * batch_size
        end_index = min((i + 1) * batch_size, total_docs)

        batch_docs = docs[start_index:end_index]

        vector_store = SupabaseVectorStore.from_documents(
            batch_docs,
            embeddings,
            client=supabase,
            table_name=os.environ.get("TABLE_NAME"),
        )

        print(f"Stored batch {i + 1}/{num_batches}")
