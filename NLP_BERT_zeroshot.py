import pandas as pd
from transformers import pipeline

import warnings
import os


# --- Configuration ---
CSV_INPUT_FILE = "dummy_records_data.csv" # Name of your input CSV file
NOTES_COLUMN_NAME = "clinical_notes"      # Column containing the text notes
OUTPUT_COLUMN_NAME = "infection_status"   # Name for the new classification column

# --- Model Selection ---
# Choose the model you want to use:
# Option 1: General purpose, robust baseline
# MODEL_NAME = "facebook/bart-large-mnli"
# Option 2: Microsoft DeBERTa but fine-tuned on MNLI (Potentially better for medical text)
# MODEL_NAME = "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli"
# Option 3: Strong general NLI model (Good benchmark, might generalize well)
MODEL_NAME = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"

print(f"Selected Model: {MODEL_NAME}")

# Define the infection status labels
infection_labels = [
    "definite infection",
    "potential infection",
    "not likely infection",
    "no infection",
]

# --- Load Data ---
print(f"Loading data from {CSV_INPUT_FILE}...")
if not os.path.exists(CSV_INPUT_FILE):
    print(f"Error: Input file not found at {CSV_INPUT_FILE}")
    print("Please ensure the CSV file is in the correct directory.")
    exit()

try:
    df = pd.read_csv(CSV_INPUT_FILE)
    print(f"Successfully loaded {len(df)} records.")
    if NOTES_COLUMN_NAME not in df.columns:
        print(f"Error: Column '{NOTES_COLUMN_NAME}' not found in the CSV file.")
        print(f"Available columns are: {list(df.columns)}")
        exit()
    df[NOTES_COLUMN_NAME] = df[NOTES_COLUMN_NAME].fillna("") # Replace NaN with empty string
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# --- Load Model ---
print(f"Loading zero-shot classification model '{MODEL_NAME}' (this may take a moment)...")
try:

    classifier = pipeline(
        "zero-shot-classification",
        model=MODEL_NAME, # Use the selected model name
    )
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model '{MODEL_NAME}': {e}")
    print("Ensure the model name is correct and you have an internet connection.")
    print("You might need to install specific dependencies for some models.")
    exit()


# --- Classification Function ---
def classify_infection_status(note_text):
    """
    Classifies a single patient note for infection status using zero-shot.
    """
    if not isinstance(note_text, str) or len(note_text.strip()) == 0:
        return "invalid input"

    try:
        result = classifier(note_text, infection_labels, multi_label=False)
        predicted_label = result['labels'][0]
        score = result['scores'][0] # You can access the confidence score
        #Optional: Print score for debugging
        print(f"Note: '{note_text[:30]}...' -> {predicted_label} ({score:.2f})")
        return predicted_label
    except Exception as e:
        print(f"Error classifying note snippet: '{str(note_text)[:50]}...' - Error: {e}")
        return "classification error"

# --- Apply Classification ---
print(f"\nClassifying notes from the '{NOTES_COLUMN_NAME}' column using {MODEL_NAME}...")

# Apply the function to the specified column
df[OUTPUT_COLUMN_NAME] = df[NOTES_COLUMN_NAME].astype(str).apply(classify_infection_status)

print("\n--- Classification Results ---")
pd.set_option('display.max_colwidth', 80)
pd.set_option('display.max_rows', 20)
print(df[[NOTES_COLUMN_NAME, OUTPUT_COLUMN_NAME]].head(15))

# Optional: Save the results to a new CSV file, including model name
try:
    # Sanitize model name for filename
    safe_model_name = MODEL_NAME.replace("/", "_")
    output_filename = f"classified_{safe_model_name}_{CSV_INPUT_FILE}"
    df.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")
except Exception as e:
    print(f"\nError saving results to CSV: {e}")


