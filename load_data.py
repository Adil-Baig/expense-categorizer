import pandas as pd

def load_and_clean_data(filepath: str) -> pd.DataFrame:
    """
    Load expense data from a CSV file and clean it for analysis.

    Steps performed:
    1. Load data from CSV into a DataFrame.
    2. Convert 'date' column to datetime (invalid dates → NaT).
    3. Drop rows with missing 'date' or 'amount'.
    4. Ensure 'amount' is stored as float.

    Args:
        filepath (str): Path to the CSV file containing expenses.

    Returns:
        pd.DataFrame: Cleaned DataFrame ready for analysis.
    """
    
    # Step 1: Load raw data
    df = pd.read_csv(filepath)

    # Step 2: Convert 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')

    # Step 3: Drop rows with missing critical fields
    df = df.dropna(subset=['date', 'amount'])

    # Step 4: Ensure 'amount' column is float
    df['amount'] = df['amount'].astype(float)

    return df


# Simple category mapping
CATEGORY_MAP = {
    "coffee": "Food & Drink",
    "groceries": "Food & Drink",
    "internet": "Utilities",
    "rent": "Housing",
    "movie": "Entertainment"
}

def categorize_expenses(df: pd.DataFrame) -> pd.DataFrame:
    """
    Categorize expenses using a simple keyword mapping.

    Args:
        df (pd.DataFrame): Cleaned expenses DataFrame.

    Returns:
        pd.DataFrame: DataFrame with an added 'category' column.
    """
    df['category'] = df['description'].str.lower().map(CATEGORY_MAP).fillna("Other")
    return df


if __name__ == "__main__":
    df = load_and_clean_data("data/sample_expenses.csv")
    df = categorize_expenses(df)
    print("\n--- Categorized expenses ---")
    print(df)


