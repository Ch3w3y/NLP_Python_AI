# test_gpu.py
from llama_cpp import Llama
import os

# --- >> UPDATE THIS PATH << ---
MODEL_PATH ="C:/Users/daryn/PycharmProjects/NLP_Python_AI/mixtral-8x7b-instruct-v0.1.Q4_K_M.gguf"

if not os.path.exists(MODEL_PATH):
    print(f"Model file not found at: {MODEL_PATH}")
    exit()

print("Attempting to load model with GPU offload...")
try:
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, verbose=True)
    print("\n*** SUCCESS: Model loaded with n_gpu_layers=-1 ***")
    # Check the verbose output above for lines like 'cuBLAS = 1'
except Exception as e:
    print(f"\n*** ERROR loading model with GPU: {e} ***")
    print("   GPU support likely not enabled during installation.")

print("\nAttempting to load model with CPU only (n_gpu_layers=0)...")
try:
    llm_cpu = Llama(model_path=MODEL_PATH, n_gpu_layers=0, verbose=False)
    print("*** SUCCESS: Model loaded with n_gpu_layers=0 (CPU) ***")
except Exception as e:
    print(f"*** ERROR loading model even on CPU: {e} ***")

