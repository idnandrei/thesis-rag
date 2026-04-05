import json
from pathlib import Path

script_dir = Path(__file__).resolve().parent
input_path = script_dir.parent / "dataset" / "longervideos" / "dataset.json"
output_path = script_dir.parent / "dataset" / "longervideos" / "lecture_dataset.json"

MAX_VIDEOS_PER_GROUP = 10

with open(input_path, "r", encoding="utf-8") as file:
    data = json.load(file)

new_data = {}
old_total = 0
new_total = 0

print(f"Keeping at most {MAX_VIDEOS_PER_GROUP} videos per lecture group.\n")

for key, entries in data.items():
    if not entries or entries[0].get("type") != "lecture":
        continue

    entry = entries[0]
    urls = entry.get("video_url", [])
    old_count = len(urls)
    kept_urls = urls[:MAX_VIDEOS_PER_GROUP]
    new_count = len(kept_urls)

    trimmed_entry = dict(entry)
    trimmed_entry["video_url"] = kept_urls

    new_data[key] = trimmed_entry

    old_total += old_count
    new_total += new_count

    print(
        f"group {key}: old={old_count}, kept={new_count}, removed={old_count - new_count}"
    )

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(new_data, file, ensure_ascii=False, indent=2)

print("\nDone.")
print(f"Old total lecture videos: {old_total}")
print(f"New total lecture videos: {new_total}")
print(f"Trimmed dataset written to: {output_path}")
