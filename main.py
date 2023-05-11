import os
from dotenv import load_dotenv
from supabase.client import Client, create_client
from langchain import LLMChain
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain.vectorstores import SupabaseVectorStore
from langchain.schema import (
    SystemMessage
)
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from load import clone_repository
from embed import embed_codebase
from analyze import analyze_and_suggest_improvements

load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

embeddings = OpenAIEmbeddings()

vector_store = SupabaseVectorStore(
    supabase, 
    embeddings, 
    table_name=os.environ.get("TABLE_NAME"),
    query_name="repo_chat_search"
)

clone_repository()

embed_codebase()

# Define the prompt template for CodeGenie AI

# Call the analyze_and_suggest_improvements function to get suggestions
analyze_and_suggest_improvements()