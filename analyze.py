import os
import openai
import re
from loguru import logger
from pyAn.analyzer import CallGraphVisitor
from dotenv import load_dotenv
from supabase import create_client
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import json
import ast 

# Load the environment variables
load_dotenv()

supabase_url = os.environ.get("SUPABASE_URL")
supabase_key = os.environ.get("SUPABASE_SERVICE_KEY")
openai.api_key = os.environ.get("OPENAI_API_KEY")
supabase = create_client(supabase_url, supabase_key)

def generate_dependency_graph(file_list):
    dependencies = {}
    for file_str in file_list:
        file_obj = json.loads(file_str)
        file_path = os.path.normpath(file_obj['source'])
        logger.info(f"Processing file: {file_path}")
        if file_path.endswith(".py"):
            with open(file_path, "r") as f:
                tree = ast.parse(f.read())
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            dependencies.setdefault(file_path, set()).add(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        for alias in node.names:
                            dependencies.setdefault(file_path, set()).add(f"{module}.{alias.name}")
        else:
            print(f"Skipping non-Python file: {file_path}")
    return dependencies


def analyze_and_suggest_improvements(batch_size=100):
    suggestions = []
    current_offset = 0

    while True:
        # Fetch only relevant columns and records in batches
        embeddings_query = (
            supabase
            .from_(os.environ.get("TABLE_NAME"))
            .select("metadata,content")
            .limit(batch_size)
        )

        result = embeddings_query.execute()

        # returns an APIResponse object, check to see if it has data
        if len(result.data) == 0:
            # no data to fetch, log this
            logger.info("No data to fetch.")
            break

        embeddings_list = result.data
        embeddings_dict = {json.dumps(embedding['metadata']): embedding for embedding in embeddings_list}
        file_list = [json.dumps(embedding['metadata']) for embedding in embeddings_list]

        # Generate dependency graph
        dependencies_dict = generate_dependency_graph(file_list)

        for embedding in embeddings_list:
            code_content = embedding['content'].strip().lstrip('\ufeff')
            file_path = os.path.normpath(embedding['metadata']['source'])

            # Find dependencies based on the dependency graph
            dependencies = dependencies_dict.get(file_path, [])

            # Combine the code content with its dependencies
            combined_code = code_content
            for dependency in dependencies:
                dep_embedding = embeddings_dict.get(json.dumps(dependency))
                if dep_embedding:
                    combined_code += f"\n\n### {dep_embedding['metadata']} ###\n{dep_embedding['content']}"

            prompt = """
                You are CodeGenie AI. You are a superintelligent AI that analyzes codebases and provides suggestions to improve or refactor code based on its underlying functionality.

                You are:
                - helpful & friendly
                - incredibly intelligent
                - an uber developer who takes pride in reviewing code and code readability

                Analyze the code snippet and its dependencies, and suggest improvements based on code optimization, best practices, and opportunities for refactoring:

                {combined_code}

                [END OF CODE FILE(S)]

                Now, provide suggestions to improve the code based on its underlying functionality.
                """
            chat = ChatOpenAI(
                streaming=True,
                callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                verbose=True,
                temperature=0.5)
            system_message_prompt = SystemMessagePromptTemplate.from_template(prompt)
            chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt])
            chain = LLMChain(llm=chat, prompt=chat_prompt)
            logger.info(f"Running GPT-4 on file {embedding['metadata']}...")
            response = chain.run(combined_code=combined_code)
            logger.info(f"Suggested improvement for file {embedding['metadata']}: {response.strip()}")

            suggestions.append({"file": embedding['metadata'], "suggestion": response.strip()})

        current_offset += batch_size

    return suggestions
