import json
import pandas as pd

def extract_features_from_json(json_file_path):
    # Read the features file
    with open(json_file_path, 'r') as f:
        data = json.load(f)
    
    # Assuming the structure has a list of features or a dictionary of importance
    # Adjust the key 'features' to match the actual key in your JSON
    # If the JSON is just a list of genes, use: features = data
    features = data.get('features', [])
    
    # Create a simple DataFrame of nodes
    df_nodes = pd.DataFrame(features, columns=['gene_name'])
    df_nodes.to_csv('nodes.csv', index=False)
    
    print(f"Extracted {len(features)} features. Saved to nodes.csv.")
    return features

if __name__ == "__main__":
    # Ensure ml_ready_features.json is in your directory
    try:
        extract_features_from_json('ml_ready_features.json')
    except FileNotFoundError:
        print("Error: ml_ready_features.json not found.")
    except Exception as e:
        print(f"An error occurred: {e}")