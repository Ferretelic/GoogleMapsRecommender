import os

from huggingface_hub import HfApi

def upload_embeddings(repo_id):
    api = HfApi()
    api.create_repo(repo_id=repo_id, exist_ok=True)

    for category in os.listdir("./embeddings"):
        for model in os.listdir(f"./embeddings/{category}"):
            if model.endswith(".pt"):
                model_path = f"./embeddings/{category}/{model}"
                path_in_repo = f"{category}/{model}"

                api.upload_file(
                    path_or_fileobj=model_path,
                    path_in_repo=path_in_repo,
                    repo_id=repo_id,
                    repo_type="model"
                )

    print("Upload has been completed")

if __name__ == "__main__":
    repo_id = "Ferretelic/google-maps-recommender"
    upload_embeddings(repo_id)