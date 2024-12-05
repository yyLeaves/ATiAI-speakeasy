import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# Path to the downloaded model
model_path = "./Llama-3.2-3B-Instruct"

# Load tokenizer and model
tokenizer = LlamaTokenizer.from_pretrained(model_path, legacy=True)
print(f"Tokenizer type after loading: {type(tokenizer)}")
model = LlamaForCausalLM.from_pretrained(model_path)
print(type(tokenizer))
# Set the model to the appropriate device (GPU or CPU)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Prepare input text
input_text = "What is the capital of France?"
print(type(tokenizer))
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Generate output
outputs = model.generate(**inputs, max_new_tokens=50)
response = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Response:", response)
