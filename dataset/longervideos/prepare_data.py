import json
import os

with open("./dataset.json", "r", encoding="utf-8") as f:
    longervideos = json.load(f)

collections = []

for _id, entries in longervideos.items():
    # entries is a list of dicts
    lecture_entries = [e for e in entries if e.get("type") == "lecture"]
    if not lecture_entries:
        continue  # skip non-lecture collections

    collection = lecture_entries[0]  # keep it simple: first lecture entry

    collection_name = f"{_id}-{collection['description']}"
    collections.append(collection_name)

    os.makedirs(os.path.join(collection_name, "videos"), exist_ok=True)

    with open(
        os.path.join(collection_name, "videos.txt"), "w", encoding="utf-8"
    ) as out:
        urls = collection.get("video_url", [])
        for i, url in enumerate(urls):
            out.write(url)
            if i != len(urls) - 1:
                out.write("\n")
