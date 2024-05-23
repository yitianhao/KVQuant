# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. LLaMA-7B model, as mentioned in their Abstract section
tokenizer = AutoTokenizer.from_pretrained("huggyllama/llama-7b")
model = AutoModelForCausalLM.from_pretrained("huggyllama/llama-7b")


