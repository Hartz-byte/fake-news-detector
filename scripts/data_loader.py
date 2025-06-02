import pandas as pd
from sklearn.utils import shuffle

def load_data(fake_path="../data/Fake.csv", true_path="../data/True.csv"):
    """
    Loads and merges the Fake and True news datasets.
    Returns a DataFrame with 'text' and 'label' columns.
    """
    # Load fake and true data
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    # Add labels: 0 for FAKE, 1 for REAL
    fake_df['label'] = 0
    true_df['label'] = 1

    # Keep only relevant columns
    fake_df = fake_df[['text', 'label']]
    true_df = true_df[['text', 'label']]

    # Concatenate and shuffle
    df = pd.concat([fake_df, true_df], ignore_index=True)
    df = shuffle(df, random_state=42).reset_index(drop=True)

    return df
