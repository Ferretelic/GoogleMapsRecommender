import os
import json

import pandas as pd
from hydra import initialize, compose

def load_category():
    with initialize(version_base=None, config_path="../config"):
        cfg = compose(config_name="base", overrides=[])
        return cfg.dataset.category

def load_meta(category):
    meta = pd.read_csv(f"./dataset/processed/{category}/combined/meta.csv")
    return meta

def search_and_select_places(meta):
    selected_gmap_ids = set()

    while True:
        print("\n--- Search Places ---")
        query = input("Enter place name to search (or 'q' to finish): ").strip()

        if not query or query.lower() == "q":
            break

        results = meta[meta["name"].str.contains(query, case=False, na=False)].reset_index(drop=True)

        if results.empty:
            print("No results found.")
            continue

        print(f"\nResults for '{query}':")
        for index, row in results.iterrows():
            print(f"[{index}] {row["name"]} ({row["state"]})")
            print(f"    Address: {row["address"]}")
            category = row["category"].replace("'", "").replace("]", "").replace("[", "").split(",")
            print(f"    Category: {",".join(category)}")

        selection = input("\nEnter indices to add (e.g., 0, 2 / Enter to skip): ").strip()

        if not selection:
            continue

        try:
            indices = [int(x.strip()) for x in selection.split(',') if x.strip().isdigit()]
            for idx in indices:
                if 0 <= idx < len(results):
                    target_id = results.loc[idx, "gmap_id"]
                    target_name = results.loc[idx, "name"]

                    selected_gmap_ids.add(target_id)
                    print(f"Added: {target_name}")
                else:
                    print(f"Invalid index: {idx}")
        except ValueError:
            print("Invalid input format.")

    return list(selected_gmap_ids)

def main():
    category = load_category()
    meta = load_meta(category)

    required_columns = {"gmap_id", "name", "address", "category", "state"}

    print("=== Create New User ===")
    os.makedirs(f"./users/{category}", exist_ok=True)
    current_count = len(os.listdir(f"./users/{category}"))
    new_user_id = current_count

    print(f"New User ID: {new_user_id}")

    user_name = input("Enter User Name: ").strip()
    if not user_name:
        print("Name is required. Aborting.")
        return

    gmap_ids = search_and_select_places(meta)
    new_user = {
        "user_id": new_user_id,
        "name": user_name,
        "gmap_ids": gmap_ids
    }

    with open(f"./users/{category}/{new_user_id}.json", "w") as f:
        json.dump(new_user, f, indent=4)

if __name__ == "__main__":
    main()