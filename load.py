import os
from dotenv import load_dotenv
from langchain.document_loaders import GitLoader

load_dotenv()

def clone_repository():
    # Get the access token from the environment variables
    access_token = os.environ.get("GITHUB_ACCESS_TOKEN")

    # Add the access token to the clone URL
    repo_url = os.environ.get("REPO_URL")
    clone_url = f"https://{access_token}@{repo_url.split('https://')[1]}"
    # clone_url=os.environ.get("REPO_URL"),

    loader = GitLoader(
        clone_url=clone_url,
        repo_path='repo',
        branch=os.environ.get("REPO_BRANCH")
    )

    print("Cloning repository...")
    if not os.path.exists('repo'):
        loader.load()
    else:
        print("Repository already exists.")

    print("Repository cloned.")
