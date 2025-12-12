import os

import requests
from tqdm import tqdm

def download_file_with_progress(url, file_name):
    response = requests.get(url, stream=True)

    response.raise_for_status()
    total_size_in_bytes = int(response.headers.get("content-length", 0))

    block_size = 1024

    print(f"Start downloading: {file_name}")

    progress_bar = tqdm(total=total_size_in_bytes, unit='iB', unit_scale=True)

    with open(file_name, "wb") as file:
        for data in response.iter_content(block_size):
            progress_bar.update(len(data))
            file.write(data)

    progress_bar.close()
    print(f"Finished downloading a file: {file_name}")

def download_dataset(raw_path):
    base_url = "https://mcauleylab.ucsd.edu/public_datasets/gdrive/googlelocal/%s-%s.json.gz"

    with open(f"{raw_path}/states.txt", "r") as f:
        states = f.read().split("\n")

    for mode in ["meta", "review"]:
        folder_path = f"{raw_path}/{mode}"
        os.makedirs(folder_path, exist_ok=True)

        for state in states:
            name = state.replace(" ", "_")
            url = base_url % (mode, name)
            file_name = f"{folder_path}/{name}.json.gz"

            if not os.path.exists(file_name):
                download_file_with_progress(url, file_name)

if __name__ == "__main__":
    raw_path = "../dataset/raw"
    download_dataset(raw_path)