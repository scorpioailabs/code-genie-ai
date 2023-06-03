import os
import re
from loguru import logger
from pyAn.analyzer import CallGraphVisitor
from dotenv import load_dotenv
from chroma_db import ChromaDB
from langchain import LLMChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
import json
import ast 
from datetime import datetime, timedelta
import time
import difflib
from suggestion import Suggestion
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("flax-community/gpt-neo-125M-code-clippy")
model = AutoModelForCausalLM.from_pretrained("flax-community/gpt-neo-125M-code-clippy")

# Load the environment variables
load_dotenv()

chroma_db_path = os.environ.get("CHROMA_DB_PATH")
chroma_db = ChromaDB(chroma_db_path)

def fetch_chroma_db_ids(batch_size=100):
    # Define an offset to paginate the results from ChromaDB
    offset = 0
    
    # Fetch ids from ChromaDB
    ids_result = chroma_db.vector_store.chroma_collection.list_ids(limit=batch_size, offset=offset)

    # Process ids_result to obtain the ids list
    id_list = [entry['id'] for entry in ids_result]

    # Update the offset for the next batch (you can decide on how to update based on your requirements).
    offset += len(id_list)

    return id_list

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

def process_code(embedding, window_size, suggestions, dependencies_dict, embeddings_dict):
    code_content = embedding['content'].strip().lstrip('\ufeff')
    success = False
    file_path = os.path.normpath(embedding['metadata']['source'])

    # Find dependencies based on the dependency graph
    dependencies = dependencies_dict.get(file_path, [])

    # Combine the code content with its dependencies
    combined_code = code_content
    for dependency in dependencies:
        dep_embedding = embeddings_dict.get(dependency)
        if dep_embedding:
            combined_code += f"\n\n### {dep_embedding['metadata']} ###\n{dep_embedding['content']}"

    # Process the combined_code with windowing
    for idx in range(0, len(combined_code), window_size):
        windowed_code = combined_code[idx:idx+window_size]

        # Add the windowed_code to the prompt
        prompt = """
            You are CodeGenie AI, a superintelligent AI that analyzes codebases ...
            (the same prompt as before)
        """
        
        input_ids = tokenizer.encode(prompt.format(windowed_code=windowed_code), return_tensors='pt')
        response = model.generate(input_ids, max_length=1000, do_sample=True, temperature=0.5)
        response_text = tokenizer.decode(response[0]).strip()

        # Check if the response does not contain the specific phrase "No improvements to be made" as a substring
        if "No improvements to be made" not in response_text:
            similarity_threshold = 0.7
            
            # Check if there are any suggestions and calculate the Jaccard similarity
            if suggestions:
                similarities = [jaccard_similarity(suggestion.suggestion, response_text) for suggestion in suggestions]
                # log
                logger.info(f"Similarities: {similarities}")
                is_duplicate = any(similarity >= similarity_threshold for similarity in similarities)

                if is_duplicate:
                    logger.info(f"Duplicate suggestion for file {embedding['metadata']}: {response_text}")
                    success = True
            else:
                is_duplicate = False
                
            # Add a new suggestion if there are no duplicates
            if not is_duplicate:
                s = Suggestion(file=str(embedding['metadata']), suggestion=response_text)
                logger.info(f"Adding suggestion for file {embedding['metadata']}:{response_text}")
                suggestions.add(s)
                logger.info(f"Suggested improvement for file {embedding['metadata']}: {response_text}")
            
            success = True  # Set success to True if the response is processed without errors

        else:
            logger.info(f"No improvements to be made for file {embedding['metadata']}")
            success = True  # Set success to True if the response is processed without errors
            continue

    # return success and suggestions
    return success, suggestions

def analyze_and_suggest_improvements(batch_size=100):
    suggestions = set()
    window_size = 2048  # You can adjust this based on your desired token count (the maximum for GPT-Neo 125M is 2048)

    logger.info("Started analyzing codebase.")

    # Customize the following loop based on how you want to fetch data (you may fetch ids or vectors from ChromaDB)
    while True:
        # Obtain the ids from ChromaDB, by modifying the fetch_chroma_db_ids function
        id_list = fetch_chroma_db_ids(batch_size)
        embeddings_list = chroma_db.get_by_id(id_list)

        # Create the embeddings_dict from the embeddings_list
        embeddings_dict = {embedding['metadata']['source']: embedding for embedding in embeddings_list}

        dependencies_dict = generate_dependency_graph(embeddings_list)

        for embedding in embeddings_list:
            success = False  # Add a flag to check if the request was successful
            while not success:  # Continue processing the current embedding until success
                success, new_suggestions = process_code(embedding, window_size, suggestions, dependencies_dict, embeddings_dict)
                if new_suggestions is not None:
                    for suggestion in new_suggestions:
                        # Use the custom Suggestion class to create a hashable object
                        s = Suggestion(file=suggestion.file, suggestion=suggestion.suggestion)
                        # Add the suggestion to the set (only if it's unique)
                        suggestions.add(s)

        # Log the number of suggestions generated
        suggestions_list = [{'file': s.file, 'suggestion': s.suggestion} for s in suggestions]
        print(f"Generated {len(suggestions_list)} suggestions in total.")

        implement_suggestions(suggestions_list)

def implement_suggestions(suggestions):
    write_suggestions_to_file(suggestions)
    
    for suggestion in suggestions:
        try:
            file_path, dictionary = get_file_path_and_dictionary(suggestion)
        except ValueError as e:
            print(e)
            continue
        try:
            original_code = read_original_code(file_path)
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error reading file {file_path}: {e}")
            continue

        updated_code = original_code
        apply_suggestion(updated_code, suggestion, file_path)

def get_file_path_and_dictionary(suggestion):
    s = suggestion['file']
    try:
        dictionary = ast.literal_eval(s)
    except ValueError as e:
        raise ValueError(f"Error parsing dictionary from suggestion: {e}")

    file_key = dictionary['source']

    # Split the file key at the last occurrence of '.git'
    repo_url, relative_file_path = file_key.rsplit('.git', 1)

    # Add the '.git' back to the repository URL
    repo_url += '.git'

    # If the relative file path starts with a '/', remove it
    if relative_file_path.startswith('/'):
        relative_file_path = relative_file_path[1:]

    return relative_file_path, dictionary

def read_original_code(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        original_code = f.read()
    return original_code

def apply_suggestion(original_code, suggestion, file_path):
    prompt = """
        You are CodeGenie AI. You are a superintelligent AI that refactors codebases based on suggestions provided by your counterpart AI to improve underlying functionality.
        You are:
        - helpful & friendly
        - incredibly intelligent
        - an uber developer who takes pride in writing code
        Utilize the following suggestion that your counterpart provided and implement them in the snippet we provide you.
        First you get the suggestion, then you get the current code window.
        Write code and nothing else, if you are writing anything in natural language then it will only be comments denoted by the appropriate syntax for the language you are writing in.
        Ultimately your goal is to implement the suggestions provided directly as code, be careful not to delete any code that is not part of the suggestions.
        Suggestions (in natural language):
        {suggestion_text}
        Code:
        {windowed_code}
        [END OF CODE FILE(S)]
    """

    # Split the original_code into windows
    window_size = 2048
    code_windows = [original_code[i:i + window_size] for i in range(0, len(original_code), window_size)]

    # Process each window using GPT-Neo
    refactored_windows = []
    for window in code_windows:
        # Pass suggestion['suggestion'] to suggestion_text instead of the whole dictionary
        input_ids = tokenizer.encode(prompt.format(suggestion_text=suggestion['suggestion'], windowed_code=window), return_tensors='pt')
        response = model.generate(input_ids, max_length=1024, do_sample=True, temperature=0.5)
        refactored_code = tokenizer.decode(response[0]).strip()
        refactored_windows.append(refactored_code)

    # Recombine the refactored windows into a single refactored_code
    refactored_code = "\n".join(refactored_windows)

    # Create an anchor map using the original_code and refactored_code
    anchor_map = create_anchor_map(original_code, refactored_code)

    # Merge the refactored code with the original_code using the anchor_map
    merged_code = merge_code(original_code.splitlines(), refactored_code.splitlines(), anchor_map)

    with open(file_path, "w") as f:
        f.write(merged_code)

def preprocess_lines(lines):
    pattern = re.compile(r"^\s*(#.*|\s*)$")
    preprocessed_lines = [line for line in lines.splitlines() if not pattern.match(line)]
    return preprocessed_lines

def find_untouched_lines(original_lines, refactored_lines):
    if not original_lines or not isinstance(original_lines, str):
        print("Invalid input for original_lines")
        return {}
    
    if not refactored_lines or not isinstance(refactored_lines, str):
        print("Invalid input for refactored_lines")
        return {}

    original_lines_list = preprocess_lines(original_lines)
    refactored_lines_list = preprocess_lines(refactored_lines)

    if not original_lines_list or not refactored_lines_list:
        print("No lines found in input")
        return {}

    matcher = difflib.SequenceMatcher(None, original_lines_list, refactored_lines_list)

    untouched_lines = {}
    for original_index, refactored_index, length in matcher.get_matching_blocks():
        if length > 0:
            for i in range(length):
                untouched_lines[original_index + i] = refactored_index + i

    return untouched_lines

def create_anchor_map(original_code, refactored_code):
    untouched_lines = find_untouched_lines(original_code, refactored_code)
    anchors = {original_idx: refactored_idx for original_idx, refactored_idx in untouched_lines.items()}
    return anchors

def merge_code(original_lines, refactored_lines, anchor_map):
    merged_lines = []
    refactored_line_idx = 0

    for original_line_idx, original_line in enumerate(original_lines):
        if original_line_idx in anchor_map:
            # If the original line should remain unchanged, add it to the merged_lines
            merged_lines.append(original_line)
        else:
            # Check if refactored_line_idx is within the valid range
            if refactored_line_idx < len(refactored_lines):
                # Otherwise, add the corresponding refactored line to the merged_lines
                merged_lines.append(refactored_lines[refactored_line_idx])
                refactored_line_idx += 1
            else:
                print(f"Warning: refactored_line_idx out of range: {refactored_line_idx}")


    # Add any remaining refactored lines to the merged_lines that were not added during the loop
    while refactored_line_idx < len(refactored_lines):
        merged_lines.append(refactored_lines[refactored_line_idx])
        refactored_line_idx += 1

    return "\n".join(merged_lines)

def write_suggestions_to_file(suggestions):
    print("Writing suggestions to file...")
    with open('suggestions.txt', 'w') as f:
        for suggestion in suggestions:
            file_path = suggestion['file']
            suggestion_text = suggestion['suggestion']
            f.write(f"File: {file_path}\n\n{suggestion_text}\n\n\n")

def apply_code_changes(original_code, code_changes):
    code_lines = original_code.splitlines()
    new_code_lines = []
    change_index = 0

    for line in code_lines:
        if change_index < len(code_changes) and code_changes[change_index]['line'] == line:
            change = code_changes[change_index]
            action = change['action']

            if action == 'remove':
                pass
            elif action == 'add':
                new_code_lines.append(change['line'])
            elif action == 'modify':
                new_code_lines.append(change['new_line'])

            change_index += 1
        else:
            new_code_lines.append(line)

    # Add any remaining 'add' actions that may not have matched any lines in the original code
    for change in code_changes[change_index:]:
        if change['action'] == 'add':
            new_code_lines.append(change['line'])

    return '\n'.join(new_code_lines)

def jaccard_similarity(str1, str2):
    set1 = set(str1.lower().split())
    set2 = set(str2.lower().split())
    intersection = set1.intersection(set2)
    union = set1.union(set2)
    return len(intersection) / len(union)
