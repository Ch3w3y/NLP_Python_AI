import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np # Import numpy for isnan check

# --- Configuration ---
csv_filename = "dummy_data_classified.csv"
true_label_col = "clinical_confirmed_infection"
predicted_label_col = "infection_status"
# Define what values represent 'Positive' (Infection) and 'Negative'
# Handles variations like TRUE/FALSE, 1/0 (as strings or numbers)
positive_values = [
    "true",
    "1",
    1,
    True,
]  # Add other variations if needed
negative_values = [
    "false",
    "0",
    0,
    False,
]  # Add other variations if needed

# --- Load Data ---
try:
    df = pd.read_csv(csv_filename)
    print(f"Successfully loaded data from '{csv_filename}'.")
except FileNotFoundError:
    print(f"Error: File '{csv_filename}' not found. Please ensure it's in the same directory or provide the full path.")
    exit()
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# --- Validate Columns ---
if true_label_col not in df.columns:
    print(f"Error: True label column '{true_label_col}' not found in the CSV.")
    exit()
if predicted_label_col not in df.columns:
    print(f"Error: Predicted label column '{predicted_label_col}' not found in the CSV.")
    exit()

# --- Preprocess Labels (Convert to consistent 0s and 1s) ---
def normalize_label(label):
    """Converts various representations (True/False, 1/0) to 1 (Positive) or 0 (Negative)."""
    label_str_lower = str(label).lower() # Convert to lowercase string for robust comparison
    if label_str_lower in [str(pv).lower() for pv in positive_values] or label in positive_values:
        return 1
    elif label_str_lower in [str(nv).lower() for nv in negative_values] or label in negative_values:
        return 0
    else:
        return np.nan # Return NaN for unrecognized values


# Apply normalization
y_true = df[true_label_col].apply(normalize_label)
y_pred = df[predicted_label_col].apply(normalize_label)

# --- Handle potential errors during normalization ---
if y_true.isnull().any() or y_pred.isnull().any():
    print("\nError: Found unrecognized values in label columns.")
    print(f"Please ensure '{true_label_col}' and '{predicted_label_col}' contain only values representing True/Positive ({positive_values}) or False/Negative ({negative_values}).")
    # Optional: Show rows with issues
    # print("\nRows with problematic values:")
    # print(df[y_true.isnull() | y_pred.isnull()])
    exit()

# Convert to integer type after ensuring no NaNs
y_true = y_true.astype(int)
y_pred = y_pred.astype(int)

print(f"\nProcessed {len(y_true)} records.")
print(f"Unique True Labels found (after normalization): {y_true.unique()}")
print(f"Unique Predicted Labels found (after normalization): {y_pred.unique()}")


# --- Calculate Confusion Matrix ---
# Ensure labels are [0, 1] for standard TN, FP, FN, TP interpretation
# If only one class exists in either true or pred, confusion_matrix might behave differently
unique_labels = sorted(pd.concat([y_true, y_pred]).unique())
if len(unique_labels) == 0:
    print("\nError: No valid labels found after processing.")
    exit()
elif len(unique_labels) == 1:
    # Handle case where only one class (0 or 1) is present overall
    # We need to force the matrix to be 2x2 for consistent TN/FP/FN/TP extraction
    # If only 0 is present, cm will be [[N, 0], [0, 0]]
    # If only 1 is present, cm will be [[0, 0], [0, N]]
    print(f"\nWarning: Only one class ({unique_labels[0]}) present in the data or predictions.")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
else:
    # Standard case with both 0 and 1 present
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]) # Explicitly set labels=[0, 1]

print("\n--- Confusion Matrix ---")
print("         Predicted No (0)  Predicted Yes (1)")
print(f"Actual No (0)    {cm[0][0]:<15} {cm[0][1]:<15}")
print(f"Actual Yes (1)   {cm[1][0]:<15} {cm[1][1]:<15}")

# --- Extract TP, TN, FP, FN ---
# Based on standard convention where matrix is:
# [[TN, FP],
#  [FN, TP]]
tn = cm[0][0]
fp = cm[0][1]
fn = cm[1][0]
tp = cm[1][1]

print("\n--- Matrix Components ---")
print(f"True Negatives (TN):  {tn} (Correctly predicted 'No Infection')")
print(f"False Positives (FP): {fp} (Incorrectly predicted 'Infection' - Type I Error)")
print(f"False Negatives (FN): {fn} (Incorrectly predicted 'No Infection' - Type II Error)")
print(f"True Positives (TP):  {tp} (Correctly predicted 'Infection')")

# --- Calculate Performance Metrics ---
# Use zero_division=0 to avoid warnings/errors if a denominator is zero
# (e.g., no positive predictions means precision is undefined/0)
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0) # Also known as Sensitivity or True Positive Rate
f1 = f1_score(y_true, y_pred, zero_division=0)

# Specificity (True Negative Rate)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

print("\n--- Performance Metrics ---")
print(f"Accuracy:    {accuracy:.4f} ((TP + TN) / Total)")
print(f"Precision:   {precision:.4f} (TP / (TP + FP)) - Out of predicted positives, how many were actual?")
print(f"Recall:      {recall:.4f} (TP / (TP + FN)) - Out of actual positives, how many were found?")
print(f"Specificity: {specificity:.4f} (TN / (TN + FP)) - Out of actual negatives, how many were found?")
print(f"F1-Score:    {f1:.4f} (Harmonic mean of Precision and Recall)")


# --- Visualize Confusion Matrix (Optional) ---
try:
    plt.figure(figsize=(7, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d", # Format as integer
        cmap="Blues",
        xticklabels=["Predicted No (0)", "Predicted Yes (1)"],
        yticklabels=["Actual No (0)", "Actual Yes (1)"],
        annot_kws={"size": 14} # Increase annotation font size
    )
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()
except ImportError:
    print("\nNote: matplotlib and seaborn are required for visualization.")
    print("Install them using: pip install matplotlib seaborn")
except Exception as e:
    print(f"\nCould not generate plot: {e}")

print("\nDone.")
