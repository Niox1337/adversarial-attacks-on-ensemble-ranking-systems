def load_json(file: str):
    import json
    import gzip

    """
    Load a JSON or JSONL (optionally compressed with gzip) file.

    Parameters:
    file (str): The path to the file to load.

    Returns:
    dict or list: The loaded JSON content. Returns a list for JSONL files, 
                  and a dict for JSON files.

    Raises:
    ValueError: If the file extension is not recognized.
    """
    if file.endswith(".json"):
        with open(file, "r") as f:
            return json.load(f)
    elif file.endswith(".jsonl"):
        with open(file, "r") as f:
            return [json.loads(line) for line in f]
    elif file.endswith(".json.gz"):
        with gzip.open(file, "rt") as f:
            return json.load(f)
    elif file.endswith(".jsonl.gz"):
        with gzip.open(file, "rt") as f:
            return [json.loads(line) for line in f]
    else:
        raise ValueError(f"Unknown file type for {file}")


def save_json(data, file: str):
    import json
    import gzip

    """
    Save data to a JSON or JSONL file (optionally compressed with gzip).

    Parameters:
    data (dict or list): The data to save. Must be a list for JSONL files.
    file (str): The path to the file to save.

    Raises:
    ValueError: If the file extension is not recognized.
    """
    if file.endswith(".json"):
        with open(file, "w") as f:
            json.dump(data, f)
    elif file.endswith(".jsonl"):
        with open(file, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    elif file.endswith(".json.gz"):
        with gzip.open(file, "wt") as f:
            json.dump(data, f)
    elif file.endswith(".jsonl.gz"):
        with gzip.open(file, "wt") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
    else:
        raise ValueError(f"Unknown file type for {file}")