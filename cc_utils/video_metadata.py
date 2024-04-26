import os
import json


class VideoMetadata:
    def __init__(self, source_path):
        self.source_path = source_path
        self.metadata_path = os.path.splitext(self.source_path)[0]
        self.metadata = None
        self.metadata_file = f"{self.metadata_path}/meta.json"
        print(self.source_path, self.metadata_file, self.metadata_path)
        # check if we already have a metadata file
        if not os.path.exists(self.metadata_file):
            self._scaffold()
        self.metadata = self.load_metadata()

    def _scaffold(self):
        self._create_folder()
        self._create_json()

    def _create_folder(self):
        if not os.path.exists(self.metadata_path):
            os.makedirs(self.metadata_path)

    # create json for metadata
    def _create_json(self):
        metadata = {
            "path": self.metadata_path,
            "original_video": {"path": self.source_path, "meta": {}},
            "resized_video": {},
            "movement_detection": {},
            "segments": [],
        }
        with open(self.metadata_file, "w") as f:
            json.dump(metadata, f)

    def load_metadata(self):
        with open(self.metadata_file, "r") as f:
            return json.load(f)

    def save_metadata(self):
        """Save the current metadata to a JSON file."""
        with open(self.metadata_file, "w") as file:
            json.dump(self.metadata, file, indent=4)

    def update_metadata(self, keys, data):
        """
        Update metadata for nested structures.
        `keys` should be a list of keys representing the path to the target dictionary.
        `data` should be a dictionary of updates to be applied at the final nested level.
        """

        def recursive_update(d, keys, data):
            key = keys[0]
            if len(keys) == 1:
                if key not in d:
                    d[key] = {}
                d[key].update(data)
            else:
                if key not in d:
                    d[key] = {}
                recursive_update(d[key], keys[1:], data)

        recursive_update(self.metadata, keys, data)
        self.save_metadata()
