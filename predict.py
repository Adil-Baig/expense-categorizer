import argparse
import joblib
import pandas as pd
import time
from pathlib import Path

def main():
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Predict expense categories using trained model.")
    parser.add_argument("--data", type=str, help="Path to CSV file containing expense descriptions.")
    parser.add_argument("--text", type=str, help="Single expense description for prediction.")
    parser.add_argument("--output", type=str, default="predictions.csv", help="Output CSV file for predictions.")

    args = parser.parse_args()

    # Load trained model and vectorizer
    model = joblib.load("models/expense_model.pkl")
    vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

    predictions = None

    if args.data:
        # Load CSV
        df = pd.read_csv(args.data)

        if "description" not in df.columns:
            raise ValueError("Input CSV must have a 'description' column.")

        X = vectorizer.transform(df["description"])
        df["predicted_category"] = model.predict(X)
        df["confidence"] = model.predict_proba(X).max(axis=1)

        # Save predictions
        output_path = Path(args.output)
        df.to_csv(output_path, index=False)
        print(f"✅ Predictions saved to {output_path}")

    elif args.text:
        X = vectorizer.transform([args.text])
        probs = model.predict_proba(X)[0]
        prediction = model.predict(X)[0]
        confidence = probs.max()

        print(f"💡 Expense: '{args.text}' → Predicted Category: {prediction} "
            f"(Confidence: {confidence:.2f})")


    else:
        raise ValueError("Please provide either --data <file.csv> or --text 'expense description'.")

    exec_time = time.time() - start_time
    print(f"⏱ Execution Time: {exec_time:.2f} seconds")

if __name__ == "__main__":
    main()

