import json
import os
import argparse

def merge_preferences(temporal_pref_path, llavahound_pref_path, save_dir):
    # Load temporal preferences
    with open(temporal_pref_path, 'r') as f:
        temporal_data = json.load(f)
    
    # Load llavahound preferences
    with open(llavahound_pref_path, 'r') as f:
        llavahound_data = [json.loads(line) for line in f]
    
    # Prepare the merged data
    merged_data = []

    # Process temporal preferences
    for entry in temporal_data:
        merged_entry = {
            "id": entry["id"],
            "video": f"normal_frames/{entry['video']}",
            "prompt": entry["prompt"],
            "answer": entry["answer"],
            "chosen": entry["chosen"],
            "chosen_score": entry["chosen_score"],
            "rejected": entry["rejected"],
            "rejected_score": entry["rejected_score"]
        }
        merged_data.append(merged_entry)

    # Process llavahound preferences
    for entry in llavahound_data:
        merged_entry = {
            "id": entry["id"],
            "video": f"sharegpt4frames/{entry['video']}",
            "prompt": entry["prompt"],
            "answer": entry["answer"],
            "chosen": entry["chosen"],
            "chosen_score": entry["chosen_score"],
            "rejected": entry["rejected"],
            "rejected_score": entry["rejected_score"]
        }
        merged_data.append(merged_entry)

    # Save the merged data
    output_path = os.path.join(save_dir, 'timewarp_explicit.json')
    with open(output_path, 'w') as f:
        json.dump(merged_data, f, indent=4)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Merge preference files into a single JSON.')
    parser.add_argument('--temporal_pref_path', type=str, help='Path to temporal_pref.json')
    parser.add_argument('--llavahound_pref_path', type=str, help='Path to llavahound_pref.jsonl')
    parser.add_argument('--save_dir', type=str, help='Directory to save the merged JSON file')

    args = parser.parse_args()

    merge_preferences(args.temporal_pref_path, args.llavahound_pref_path, args.save_dir)