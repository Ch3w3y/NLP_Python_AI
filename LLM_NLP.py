import pandas as pd
import warnings
import os
import time # Import time for basic benchmarking
from llama_cpp import Llama # Import the Llama class

# --- Configuration ---
# --- >> 1. SET YOUR CSV FILE PATH << ---
CSV_INPUT_FILE = "dummy_records_data.csv" # Your 1000-row CSV file
NOTES_COLUMN_NAME = "clinical_notes"      # Column with the notes
OUTPUT_COLUMN_NAME = "llm_infection_status" # Column for LLM results

# --- >> 2. SET THE PATH TO YOUR DOWNLOADED GGUF MODEL << ---
MODEL_PATH = "mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf" # <--- EXAMPLE PATH! UPDATE THIS!

# --- >> 3. CONFIGURE LLAMA SETTINGS (Adjust if needed) << ---
N_GPU_LAYERS = -1    # -1 means offload all possible layers to GPU (recommended for 3090)
N_CTX = 4096         # Context window size (Mixtral supports larger context)
MAX_TOKENS_GEN = 25  # Max tokens for the label output (adjust if needed)
TEMPERATURE = 0.1    # Low temperature for deterministic classification

# Define the infection status labels (for the prompt)
infection_labels = [
    "definite infection",
    "evidence of infection",
    "no evidence of infection",
]
label_string = ", ".join([f"'{label}'" for label in infection_labels])

# --- Load Data ---
print(f"Loading data from {CSV_INPUT_FILE}...")
if not os.path.exists(CSV_INPUT_FILE):
    print(f"Error: Input file not found at {CSV_INPUT_FILE}")
    exit()
try:
    df = pd.read_csv(CSV_INPUT_FILE)
    # Ensure we only process the first 1000 rows if the file is larger
    if len(df) > 1000:
        print(f"File has {len(df)} rows. Processing the first 1000.")
        df = df.head(1000)
    else:
         print(f"Successfully loaded {len(df)} records.")

    if NOTES_COLUMN_NAME not in df.columns:
        print(f"Error: Column '{NOTES_COLUMN_NAME}' not found.")
        exit()
    df[NOTES_COLUMN_NAME] = df[NOTES_COLUMN_NAME].fillna("")
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# --- Load Local LLM ---
print(f"Loading LLM from {MODEL_PATH}...")
if not os.path.exists(MODEL_PATH):
    print(f"Error: Model file not found at {MODEL_PATH}")
    print("Please download the GGUF model file and update MODEL_PATH.")
    exit()

try:
    start_load_time = time.time()
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=N_GPU_LAYERS,
        n_ctx=N_CTX,
        verbose=True # Set to True for detailed llama.cpp output during loading/inference
    )
    end_load_time = time.time()
    print(f"LLM loaded successfully in {end_load_time - start_load_time:.2f} seconds.")
except Exception as e:
    print(f"Error loading LLM: {e}")
    print("Ensure llama-cpp-python is installed correctly with GPU support (cuBLAS).")
    exit()

# --- Prompt Engineering ---
# This prompt structure works well for Mixtral Instruct.
# You might need to refine it based on observed results.
def create_prompt(note_text):
    system_message = f"""You are an expert clinical assistant analyzing patient notes. Your task is to classify the likelihood of infection based ONLY on the provided note. Choose the single best label from the following options: {label_string}. Output ONLY the chosen label and nothing else. Do not add explanations, justifications, or any surrounding text."""

    user_message = f"""Patient Note:
---
{note_text}
---
Classification Label:"""

    # Mixtral Instruct format
    prompt = f"<s>[INST] {system_message}\n\n{user_message} [/INST]"
    return prompt

# --- Classification Function using LLM ---
def classify_with_llm(note_text):
    if not isinstance(note_text, str) or len(note_text.strip()) == 0:
        return "invalid input"

    # Truncate very long notes if they exceed context window (simple approach)
    # A more sophisticated approach would use a sliding window or summarization
    max_note_length_for_prompt = N_CTX - 256 # Reserve space for prompt template/output
    if len(note_text) > max_note_length_for_prompt:
        note_text = note_text[:max_note_length_for_prompt] + "... (truncated)"

    prompt = create_prompt(note_text)

    try:
        response = llm.create_completion(
            prompt=prompt,
            max_tokens=MAX_TOKENS_GEN,
            temperature=TEMPERATURE,
            stop=["\n", "</s>", "[INST]", "User:", "System:"] # Stop generation early
        )
        predicted_text = response['choices'][0]['text'].strip().lower()

        # --- Output Parsing Logic ---
        # Attempt to find an exact match first (case-insensitive)
        parsed_label = "parsing error" # Default if no match found
        for label in infection_labels:
            # Check if the cleaned output *is* one of the labels
            # Remove potential quotes or extra characters common in LLM outputs
            cleaned_output = predicted_text.replace("'", "").replace('"', '').strip()
            if cleaned_output == label:
                parsed_label = label
                break # Found exact match

        # If no exact match, check if a label is contained (less reliable)
        if parsed_label == "parsing error":
             for label in infection_labels:
                 if label in predicted_text:
                     # Add a basic length check to avoid partial matches in verbose output
                     if len(predicted_text) < len(label) + 15:
                         parsed_label = label
                         # print(f"Warning: Used substring match for output: '{predicted_text}' -> '{label}'")
                         break

        if parsed_label == "parsing error":
             print(f"Warning: Could not parse valid label from LLM output: '{predicted_text}' for note: '{note_text[:50]}...'")

        return parsed_label

    except Exception as e:
        print(f"Error during LLM inference for note: '{note_text[:50]}...' - {e}")
        return "llm inference error"

# --- Apply Classification ---
print(f"\nClassifying {len(df)} notes using local LLM ({os.path.basename(MODEL_PATH)})...")
start_classify_time = time.time()

# Apply the function row by row
df[OUTPUT_COLUMN_NAME] = df[NOTES_COLUMN_NAME].astype(str).apply(classify_with_llm)

end_classify_time = time.time()
total_time = end_classify_time - start_classify_time
avg_time_per_note = total_time / len(df) if len(df) > 0 else 0

print(f"\nClassification finished in {total_time:.2f} seconds ({avg_time_per_note:.3f} seconds per note on average).")

# --- Display Results ---
print("\n--- LLM Classification Results Sample ---")
pd.set_option('display.max_colwidth', 80)
pd.set_option('display.max_rows', 20)
# Show counts of each classification
print("\nClassification Counts:")
print(df[OUTPUT_COLUMN_NAME].value_counts())
# Show head of results
print("\nSample Rows:")
print(df[[NOTES_COLUMN_NAME, OUTPUT_COLUMN_NAME]].head(15))

# --- Save Results ---
try:
    safe_model_name = os.path.basename(MODEL_PATH).replace(".gguf", "").replace(".","") # Sanitize name
    output_filename = f"llm_classified_{safe_model_name}_{os.path.basename(CSV_INPUT_FILE)}"
    df.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")
except Exception as e:
    print(f"\nError saving results to CSV: {e}")

print("\n--- Next Steps ---")
print("1. Review the output CSV carefully. How accurate are the classifications?")
print("2. Refine the prompt in `create_prompt` if results are poor or inconsistent.")
print("3. Adjust the parsing logic in `classify_with_llm` if labels aren't extracted correctly.")
print("4. Consider trying a different model or quantization level if needed.")
