import os
import json


def scaffold(source_path):
    metadata_path = os.path.splitext(source_path)[0]
    create_folder(metadata_path)
    create_json(metadata_path, source_path)


# these should probably be private methods
def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)


# create json for metadata
def create_json(metadata_folder, source_path):
    metadata = {"source_path": source_path}
    with open(f"{metadata_folder}/meta.json", "w") as f:
        json.dump(metadata, f)
