import os

from huggingface_hub import HfApi

def upload_embeddings(repo_id):
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)

    api.upload_folder(
        folder_path="./embeddings",
        repo_id=repo_id,
        path_in_repo="",
        repo_type="model"
    )

    print("Upload has been completed")

if __name__ == "__main__":
    repo_id = "Ferretelic/google-maps-recommender"
    upload_embeddings(repo_id)