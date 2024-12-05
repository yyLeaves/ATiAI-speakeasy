import torch
from transformers import pipeline

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.set_num_threads(1)

torch.manual_seed(42)

# Check for MPS availability and set the device accordingly
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# Path to your downloaded Llama model
model_id = "./Llama-3.2-3B-Instruct"

# Create the pipeline with adjusted settings
pipe = pipeline(
    "text-generation",
    model=model_id,
    device=device,        # Use MPS if available
    # do_sample=False,      # Disable sampling for deterministic output
    torch_dtype=torch.float32,
    temperature=0.01,      # Set temperature to default
    top_k=None,           # Remove top_k parameter
    top_p=None,           # Remove top_p parameter
)

# Example structured message to extract movie titles directly
messages = [
    {"role": "system","content": "Extract the movie titles from the user's input and list them, one per line, with no additional text."},    # {"role": "system","content": "You are tasked to only list the movie entities from the input, ensuring no additional content. Output each movie title in a new line and not add any extra signs."}
    {"role": "user", "content": "Sentence to extract: Given that I like The Lion King, Pocahontas, and The Beauty and the Beast, can you recommend some movies?"},
]

# Generate the response
outputs = pipe(messages, max_new_tokens=100)
print(outputs)
# print(outputs)
# Loop through the outputs to find the 'assistant' role and extract its content
for message in outputs[0]['generated_text']:  # Navigate through the list of role messages
    if message['role'] == 'assistant':
        ner_output = message['content']
        break

# Check if content was found and print it
if ner_output:
    print(ner_output)
else:
    print("")

from difflib import SequenceMatcher

def filter_titles(extracted_titles, input_text, threshold=0.6):
    valid_titles = []
    input_text_lower = input_text.lower()
    words_in_input = input_text_lower.split()
    for title in extracted_titles:
        title = title.strip()
        if title:
            title_lower = title.lower()
            title_processed = preprocess_title(title_lower)
            found_match = False
            # Compare the extracted title with parts of the input text
            for i in range(len(words_in_input)):
                for j in range(i+1, len(words_in_input)+1):
                    input_fragment = ' '.join(words_in_input[i:j])
                    input_fragment_processed = preprocess_title(input_fragment)
                    similarity = SequenceMatcher(None, title_processed, input_fragment_processed).ratio()
                    if similarity >= threshold:
                        valid_titles.append(title)
                        found_match = True
                        break
                if found_match:
                    break  # Break the outer loop if a match is found
    return valid_titles

def preprocess_title(title):
    # Remove leading articles
    articles = ['a ', 'an ', 'the ']
    for article in articles:
        if title.startswith(article):
            title = title[len(article):]
            break
    return title.strip()

# Use the function after your existing code
if ner_output:
    # Get the user input text
    input_text = messages[1]['content']

    # Split the ner_output into lines (assuming each title is on a new line)
    extracted_titles = ner_output.strip().split('\n')

    # Filter the extracted titles
    valid_titles = filter_titles(extracted_titles, input_text)

    # Print the valid movie titles
    if valid_titles:
        print("Extracted Movie Titles:")
        for title in valid_titles:
            print(title)
    else:
        print("No valid movie titles found in the input.")
else:
    print("No movie titles extracted.")
