import json

def inspect_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
            print("File loaded successfully.")
            print("Structure type:", type(data))
            print("Content preview:", data)
    except Exception as e:
        print(f"Error reading file: {e}")

if __name__ == "__main__":
    inspect_json('ml_ready_features.json')