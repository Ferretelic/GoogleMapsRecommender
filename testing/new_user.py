import os
import json

from utils import *

def load_new_users(cfg):
    users = []
    for file_name in os.listdir(cfg.paths.users):
        with open(f"{cfg.paths.users}/{file_name}", "r") as f:
            user = json.load(f)
        users.append(user)

    return users

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
            category_list = row["category"].replace("'", "").replace("]", "").replace("[", "").split(",")
            print(f"[{index}] {row["name"]} ({row["state"]})")
            print(f"    Address: {row["address"]}")
            print(f"    Category: {",".join(category_list)}")

        selection = input("\nEnter indices to add (e.g., 0, 2 / Enter to skip): ").strip()

        if not selection:
            continue

        try:
            indices = [int(x.strip()) for x in selection.split(",") if x.strip().isdigit()]
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

def add_new_user(cfg):
    meta = load_combined_datast(cfg, "meta")
    os.makedirs(cfg.paths.users, exist_ok=True)

    while True:
        print("\n" + "=" * 20)
        print("   Create New User")
        print("=" * 20)

        current_count = len(os.listdir(cfg.paths.users))
        new_user_id = current_count

        print(f"New User ID: {new_user_id}")

        user_name = input("Enter User Name (or 'q' to quit): ").strip()
        if user_name.lower() == "q":
            print("Exiting...")
            break

        if not user_name:
            print("Name is required. Skipping...")
            continue

        gmap_ids = search_and_select_places(meta)

        new_user = {
            "user_id": new_user_id,
            "name": user_name,
            "gmap_ids": gmap_ids
        }

        save_path = f"{cfg.paths.users}/{new_user_id}.json"
        with open(save_path, "w") as f:
            json.dump(new_user, f, indent=4)

        print(f"\nSuccessfully saved user '{user_name}' (ID: {new_user_id}) to {save_path}")

        cont = input("\nCreate another user? (y/n): ").strip().lower()
        if cont != "y":
            print("Finished.")
            break