from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Correct path to the model directory
model_path = "models/translation"  # Replace with your actual local model path

# Load the tokenizer and model from the local path
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

# Confirm if tokenizer and model are loaded correctly
print(f"Tokenizer: {tokenizer}")
print(f"Model: {model}")
