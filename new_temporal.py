import json
import time

def process_frame_swapping_data(filepath):
    with open(filepath, "r") as f:
        data = json.load(f)

    # Delete all entries where the "type" is "subtask1"
    data = [entry for entry in data if entry["type"] != "subtask1"]

    assert all(entry["type"] != "subtask1" for entry in data), "There are still entries with type 'subtask1'"

    # Select either 5k samples or all samples if they are less than 5k
    if len(data) > 5000:
        import random
        random.seed(42)
        data = random.sample(data, 5000)

    # Change the prompt for "subtask2"
    original_chosen_list = []
    for entry in data:
        if entry["type"] == "subtask2":
            chosen = entry["chosen"]
            original_chosen_list.append(chosen)
            prompt = entry["prompt"]
            events_start = prompt.find("LIST OF EVENTS")
            if events_start != -1:
                events = prompt[events_start:]
                events = events.replace("\n", " ")
                events = ' '.join(events.split())
                new_prompt = "What is the correct order of the following events as it occurs in the video?\n" + events
                entry["prompt"] = new_prompt

    # Save the original chosen responses list to a json file
    original_chosen_output_filepath = "original_chosen_list.json"
    with open(original_chosen_output_filepath, "w") as f:
        json.dump(original_chosen_list, f, indent=4)
    print(f"Original chosen responses saved to {original_chosen_output_filepath}")

    import random

    for entry in data:
        # Change the "video" key to "id"
        entry["id"] = entry.pop("video")

        # Delete the "type"
        if "type" in entry:
            del entry["type"]

        # Add "answer" whose value is same as "chosen"
        entry["answer"] = entry["chosen"]

        # Add "chosen_score" which is a random integer sampled from 3, 5 (both included)
        entry["chosen_score"] = random.randint(3, 5)

        # Add "rejected_score" which is a random integer sampled from 0, 3 (both included)
        entry["rejected_score"] = random.randint(0, 3)

    # Assert that the length of the dataset is 5k
    assert len(data) <= 5000, "The length of the dataset is > 5k"

    # Remove any duplicate entries
    seen_entries = set()
    unique_data = []
    for entry in data:
        entry_str = json.dumps(entry, sort_keys=True)
        if entry_str not in seen_entries:
            unique_data.append(entry)
            seen_entries.add(entry_str)
    data = unique_data

    # Assert that there are no duplicate entries
    unique_entries = {json.dumps(entry, sort_keys=True) for entry in data}
    assert len(unique_entries) == len(data), "There are duplicate entries in the dataset"

    print("Final length of the dataset: ", len(data))

    # Save the new json format
    output_filepath = "processed_frame_swapping_pref_new.json"
    with open(output_filepath, "w") as f:
        json.dump(data, f, indent=4)
    print(f"Processed data saved to {output_filepath}")


process_frame_swapping_data("/Users/sameepvani/Desktop/CVPR/frame_swapping_pref_new.json")