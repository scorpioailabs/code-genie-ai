import os
from dotenv import load_dotenv
from github import Github
import subprocess
import shutil
import stat

load_dotenv()

def clone_repository():
    # Get the access token from the environment variables
    access_token = os.getenv("GITHUB_ACCESS_TOKEN")
    if not access_token:
        raise Exception("Missing GITHUB_ACCESS_TOKEN environment variable")

    g = Github(access_token)

    # Check if the repo URL is provided
    repo_url = os.getenv("REPO_URL")
    if not repo_url:
        raise Exception("Missing REPO_URL environment variable")

    # Check if the local path is provided
    local_path = os.getenv("LOCAL_PATH")
    if not local_path:
        raise Exception("Missing LOCAL_PATH environment variable")

    # Add the access token to the clone URL
    clone_url = f"https://{access_token}@{repo_url.split('https://')[1]}"

    new_branch_name = "codegenieai-feature-branch"

    print("Preparing for clone...")

    # If the local_path exists and is not the intended repository, delete it
    original_working_directory = os.getcwd()

    # Check if the local path exists
    if os.path.exists(local_path):
        # Change the working directory to the local path
        os.chdir(local_path)

        # Change the working directory back to the original one
        os.chdir(original_working_directory)

        print(f"repo_url: {repo_url}")

        # check if the local path contains the same repository as the one provided
      

        # Print the current repository URL
        print(f"current_repo_url: {current_repo_url}")

        if current_repo_url != repo_url:
            print("Different repository detected in the local path. Removing the path.")
            remove_directory(local_path)
        else:
            print("Repository already exists. Updating the repository.")
            subprocess.check_call(['git', '-C', local_path, 'pull'])

    if not os.path.exists(local_path):
        # Clone the repository
        subprocess.check_call(['git', 'clone', clone_url, local_path])
    
    # Change directory to the cloned repository before checking out a new branch
    os.chdir(local_path)

    # Check if the branch already exists
    branches = subprocess.check_output(['git', 'branch']).decode().split('\n')
    if new_branch_name in branches:
        print(f"Branch {new_branch_name} already exists. Checking it out.")
        subprocess.check_call(['git', 'checkout', new_branch_name])
    else:
        # Create and checkout a new branch
        subprocess.check_call(['git', 'checkout', '-b', new_branch_name])

    print("Repository cloned and new bSranch checked out.")

def handle_remove_readonly(func, path, exc):
    excvalue = exc[1]
    if func in (os.rmdir, os.remove) and excvalue.errno == errno.EACCES:
        os.chmod(path, stat.S_IRWXU| stat.S_IRWXG| stat.S_IRWXO) # 0777
        func(path)
    else:
        raise

def remove_directory(relative_path):
    shutil.rmtree(relative_path, ignore_errors=False, onerror=handle_remove_readonly)
