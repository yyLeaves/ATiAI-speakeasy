import torch
from transformers import pipeline

# Set the device to MPS
device = "mps" if torch.has_mps else "cpu"
print(device)

# Initialize the pipeline
model_id = "./Llama-3.2-3B-Instruct"  # Adjust path if necessary
pipe = pipeline(
    "text-generation",
    model=model_id,
    torch_dtype=torch.float32,  # Use float32 for MPS compatibility
    device=device,
)

# Define messages
messages = [
    {"role": "system", "content": "You are a pirate chatbot who always responds in pirate speak!"},
    {"role": "user", "content": "Who are you?"},
]

# Generate output
outputs = pipe(
    messages,
    max_new_tokens=256,
)
print(outputs[0]["generated_text"])