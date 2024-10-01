import pandas as pd
import os
import json
from tqdm import tqdm

DATA_DIR = '/scratch/svani/data/finevideo/finevideo_15k'
ANNOTATION_DIR = '/scratch/svani/data/finevideo'
ANNOTATION_FILENAME='finevideo-15k-descriptions.json'

def generate_annotations(parquet_filename):
    file = pd.read_parquet(os.path.join(DATA_DIR, parquet_filename))
    annotations = file['json'] # contains 33 annotations
    descriptions = {}

    # For each json file
    for i, _json in enumerate(annotations):
        # Initialize description / caption of a video
        description = "Short Description: "

        # Add short description / overall description of the video
        description += _json['content_metadata']['description'] + "\nDetailed Description: First, "

        # Add detailed activity description
        for scene in _json['content_metadata']['scenes']:
            # Add activity description
            for index, activity in enumerate(scene['activities']):
                if index == len(scene['activities']) - 1:
                    description += activity['description'] + ' '
                else:
                    description += activity['description'] + " then "
            
            # Add Narrative Progression
            for np in scene['narrativeProgression']:
                description += np['description'] + ' '

        descriptions[parquet_filename + str(i)] = description

    return descriptions

if __name__ == "__main__":
    descriptions = {}
    for file_name in tqdm(os.listdir(DATA_DIR)):
        descriptions.update(generate_annotations(file_name))
    
    with open(os.path.join(ANNOTATION_DIR, ANNOTATION_FILENAME), 'w') as f:
        json.dumps(descriptions, f)