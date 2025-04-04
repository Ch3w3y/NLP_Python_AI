import pandas as pd
import warnings
import os
import time
import torch # PyTorch is needed for transformers
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)

# Suppress specific warnings if needed
warnings.filterwarnings("ignore")

# --- Configuration ---
# --- >> 1. SET YOUR CSV FILE PATH << ---
CSV_INPUT_FILE = "dummy_records_data.csv" # Your 1000-row CSV file
NOTES_COLUMN_NAME = "clinical_notes"      # Column with the notes
OUTPUT_COLUMN_NAME = "llm_infection_status" # Column for LLM results

# --- >> 2. SET THE HUGGING FACE MODEL ID << ---
# Using Mixtral Instruct - requires trust_remote_code=True
model_id = "mistralai/Mistral-7B-Instruct-v0.2"

# --- >> 3. CONFIGURE TRANSFORMERS/GENERATION SETTINGS << ---
# Context window size - check model's max length, but use this for truncation
# Mixtral has a large context, but keep it reasonable for performance/memory
N_CTX = 4096
# Max tokens for the classification label output (keep it small)
MAX_NEW_TOKENS = 25
# Low temperature for more deterministic classification
TEMPERATURE = 0.1
# Set to False for classification tasks to be more deterministic
DO_SAMPLE = False

# Define the infection status labels (for the prompt and parsing)
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
    if len(df) > 1000:
        print(f"File has {len(df)} rows. Processing the first 1000.")
        df = df.head(1000)
    else:
         print(f"Successfully loaded {len(df)} records.")

    if NOTES_COLUMN_NAME not in df.columns:
        print(f"Error: Column '{NOTES_COLUMN_NAME}' not found.")
        exit()
    # Ensure notes column is string and handle potential NaN values
    df[NOTES_COLUMN_NAME] = df[NOTES_COLUMN_NAME].fillna("").astype(str)
except Exception as e:
    print(f"Error reading CSV file: {e}")
    exit()

# --- Load Local LLM using Transformers with 4-bit Quantization ---
print(f"Loading LLM: {model_id} (using 4-bit quantization)...")

# Configure 4-bit quantization
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 on RTX 3090
    bnb_4bit_use_double_quant=True,
)

try:
    start_load_time = time.time()

    # Load the quantized model
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto", # Automatically uses CUDA if available
        trust_remote_code=True, # REQUIRED for Mixtral
    )

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    # Set pad token if it's not set (common fix)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

    end_load_time = time.time()
    print(f"LLM and Tokenizer loaded successfully in {end_load_time - start_load_time:.2f} seconds.")

    # Create the text generation pipeline
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        # We pass generation parameters during the pipe call later
    )

except Exception as e:
    print(f"Error loading LLM using Transformers: {e}")
    print("Ensure PyTorch, transformers, accelerate, and bitsandbytes are installed correctly.")
    print("Make sure your CUDA setup in WSL2 is working (`nvidia-smi`).")
    exit()


# --- Prompt Engineering Definition (using messages for chat template) ---
# System message defining the task and desired output format
system_message_content = f"""You are an expert clinical assistant analyzing patient notes. Your task is to classify the likelihood of infection based ONLY on the provided note. Choose the single best label from the following options: {label_string}. Output ONLY the chosen label and nothing else. Do not add explanations, justifications, or any surrounding text. Please do not worry about our interpretation of the label we will be using this output to highlight most likely infections but this is not for clinical management it is purely a surveillance exercise."""

# --- Classification Function using Transformers Pipeline ---
def classify_with_llm(note_text):
    if not isinstance(note_text, str) or len(note_text.strip()) == 0:
        return "invalid input"

    # --- Truncate long notes (based on estimated token count, simple approach) ---
    # Tokenize to estimate length, truncate if needed.
    # This is approximate; a more precise method would handle token limits better.
    # Reserve some tokens for the prompt template and output.
    max_input_tokens = N_CTX - MAX_NEW_TOKENS - 100 # Conservative buffer
    inputs = tokenizer(note_text, return_tensors="pt", truncation=False)
    if inputs['input_ids'].shape[1] > max_input_tokens:
        # Simple truncation based on tokens
        truncated_ids = inputs['input_ids'][0, :max_input_tokens]
        note_text = tokenizer.decode(truncated_ids, skip_special_tokens=True) + "... (truncated)"
        # print(f"Note truncated to approx {max_input_tokens} tokens.") # Optional debug

    # --- Create Prompt using Chat Template ---
    # Define the messages for the chat template
    messages = [
        {"role": "system", "content": system_message_content},
        {"role": "user", "content": f"Patient Note:\n---\n{note_text}\n---\nClassification Label:"}
    ]
    # apply_chat_template handles the specific [INST] tags etc.
    prompt = pipe.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    try:
        # --- Perform Inference ---
        outputs = pipe(
            prompt,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=DO_SAMPLE,
            temperature=TEMPERATURE,
            # top_p=None, # Not needed if do_sample=False
            # top_k=None, # Not needed if do_sample=False
            return_full_text=False, # Ask pipeline to return only generated text
            # eos_token_id=tokenizer.eos_token_id, # Pipeline usually handles this
            # pad_token_id=tokenizer.pad_token_id
        )
        # Extract the generated text (pipeline should return only new text with return_full_text=False)
        predicted_text = outputs[0]['generated_text'].strip().lower()

        # --- Output Parsing Logic (same as before) ---
        parsed_label = "parsing error" # Default if no match found
        for label in infection_labels:
            cleaned_output = predicted_text.replace("'", "").replace('"', '').strip()
            if cleaned_output == label:
                parsed_label = label
                break # Found exact match

        if parsed_label == "parsing error":
             for label in infection_labels:
                 if label in predicted_text:
                     if len(predicted_text) < len(label) + 15: # Basic length check
                         parsed_label = label
                         # print(f"Warning: Used substring match for output: '{predicted_text}' -> '{label}'")
                         break

        if parsed_label == "parsing error":
             print(f"Warning: Could not parse valid label from LLM output: '{predicted_text}' for note: '{note_text[:50]}...'")

        return parsed_label

    except Exception as e:
        print(f"Error during LLM inference for note: '{note_text[:50]}...' - {e}")
        # Add more specific error details if possible
        if "CUDA out of memory" in str(e):
            print("CUDA out of memory error. Try reducing N_CTX or using a smaller model/quantization.")
        return "llm inference error"

# --- Apply Classification ---
print(f"\nClassifying {len(df)} notes using Transformers LLM ({model_id})...")
start_classify_time = time.time()

# Apply the function row by row
# Ensure the column is string type before applying
df[OUTPUT_COLUMN_NAME] = df[NOTES_COLUMN_NAME].apply(classify_with_llm)

end_classify_time = time.time()
total_time = end_classify_time - start_classify_time
avg_time_per_note = total_time / len(df) if len(df) > 0 else 0

print(f"\nClassification finished in {total_time:.2f} seconds ({avg_time_per_note:.3f} seconds per note on average).")

# --- Display Results ---
print("\n--- LLM Classification Results Sample ---")
pd.set_option('display.max_colwidth', 80)
pd.set_option('display.max_rows', 20)
print("\nClassification Counts:")
print(df[OUTPUT_COLUMN_NAME].value_counts())
print("\nSample Rows:")
print(df[[NOTES_COLUMN_NAME, OUTPUT_COLUMN_NAME]].head(15))

# --- Save Results ---
try:
    # Sanitize model name for filename
    safe_model_name = model_id.replace("/", "_").replace("-", "").replace(".","")
    output_filename = f"llm_classified_{safe_model_name}_{os.path.basename(CSV_INPUT_FILE)}"
    df.to_csv(output_filename, index=False)
    print(f"\nResults saved to {output_filename}")
except Exception as e:
    print(f"\nError saving results to CSV: {e}")

# Clean up GPU memory
import gc
del model
del pipe
gc.collect()
torch.cuda.empty_cache()
print("\nCleaned up GPU resources.")
