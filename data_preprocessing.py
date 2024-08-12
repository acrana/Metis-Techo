# defunct for now

import pandas as pd

def load_data(file_path):
    """Load the dataset from a CSV file."""
    return pd.read_csv(file_path)

def preprocess_data(data):
    """Preprocess the dataset (e.g., handle missing values)."""
    data = data.dropna()
    return data

def save_preprocessed_data(data, output_path):
    """Save the preprocessed dataset to a CSV file."""
    data.to_csv(output_path, index=False)

# data paths need updating

if __name__ == "__main__":
    input_path = "data/simpledata.csv"
    output_path = "data/preprocessed_simpledata.csv"

    # Load and preprocess data
    data = load_data(input_path)
    preprocessed_data = preprocess_data(data)

    # Save the preprocessed data
    save_preprocessed_data(preprocessed_data, output_path)

    print("Data preprocessing completed successfully.")
