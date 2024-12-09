import torch
from transformers import pipeline
from difflib import SequenceMatcher

# Global Torch settings to ensure deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.manual_seed(42)

# # Check for MPS availability and set the device accordingly
# device = "mps" if torch.backends.mps.is_available() else "cpu"
# print(f"Using device: {device}")

# Path to your downloaded Llama model
# model_id = "./Llama-3.2-3B-Instruct"

class LlamaProcessor:
    _instance = None
    _pipe = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(LlamaProcessor, cls).__new__(cls)
            print("Initializing LlamaProcessor instance")  # This print will only show once

            # Ensure the model is loaded only once
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            print(f"Using device: {device}")
            model_id = "./Llama-3.2-3B-Instruct"
            cls._pipe = pipeline(
                "text-generation",
                model=model_id,
                device=device,  # Use MPS if available
                torch_dtype=torch.float32,
                temperature=0.01,  # Set temperature to default
                top_k=None,        # Remove top_k parameter
                top_p=None         # Remove top_p parameter
            )
            print("Pipeline initialized")
        return cls._instance
    
    def __init__(self):
        # The pipe is already initialized in __new__
        self.pipe = self._pipe

    def preprocess_title(self, title):
        """
        Preprocess titles by removing leading articles.
        """
        articles = ['a ', 'an ', 'the ']
        title = title.strip()
        for article in articles:
            if title.startswith(article):
                title = title[len(article):]
                break
        return title.strip()

    def filter_titles(self, extracted_titles, input_text, threshold=0.6):
        """
        Filter the extracted titles based on similarity with the input text.
        """
        valid_titles = set()  # Use a set to avoid duplicates
        input_text_lower = input_text.lower()
        words_in_input = input_text_lower.split()

        for title in extracted_titles:
            title = title.strip()
            if title:
                # Split the title by the last comma and only keep the entity name
                if ',' in title:
                    entity, _ = title.rsplit(',', 1)
                    entity = entity.strip()
                else:
                    entity = title

                entity_lower = entity.lower()
                entity_processed = self.preprocess_title(entity_lower)
                found_match = False

                # Compare the entity with parts of the input text
                for i in range(len(words_in_input)):
                    for j in range(i + 1, len(words_in_input) + 1):
                        input_fragment = ' '.join(words_in_input[i:j])
                        input_fragment_processed = self.preprocess_title(input_fragment)
                        similarity = SequenceMatcher(None, entity_processed, input_fragment_processed).ratio()

                        if similarity >= threshold:
                            valid_titles.add(entity)  # Keep only the entity name
                            found_match = True
                            break  # Break the inner loop
                    if found_match:
                        break  # Break the outer loop
        return list(valid_titles)

    def extract_entities(self, input_text):
        """
        Extract entities using the defined pipeline and process the output.

        Args:
            input_text (str): The input text to extract entities from.

        Returns:
            list: Valid extracted entities after filtering.
        """
        # Example structured message to extract movie titles directly
        messages = [
            {"role": "system","content": "Extract all entities from the user's input, including movie titles, directors, actors, and any other relevant information. "
                    "List each entity on a new line, <entity, type>, (e.g., 'The Lion King, Title', 'Jon Favreau, Director', 'Meryl Streep, Actor'). Do not worry about finding any connection to the items, just list them."
                    "Do not include any additional text. Only names you know!"},
            {"role": "user", "content": "Sentence to extract: " + input_text},
        ]

        # Debugging message to see the exact messages sent to the pipeline
        # print("DEBUG: Messages being sent to pipeline:", messages)

        # Generate the response
        outputs = self.pipe(messages, max_new_tokens=100)

        # Debugging message to see the output from the pipeline
        # print("DEBUG: Outputs received from pipeline:", outputs)

        ner_output = ""

        # Extract the content from the assistant's response
        if isinstance(outputs, list) and len(outputs) > 0 and 'generated_text' in outputs[0]:
            for message in outputs[0]['generated_text']:
                # print("DEBUG: Processing message in output:", message)  # Debugging message
                if isinstance(message, dict) and message.get('role') == 'assistant':
                    ner_output = message.get('content', "")
                    break

        # Debugging message to see what was extracted
        # print("DEBUG: Extracted ner_output:", ner_output)

        # Check if content was found
        if not ner_output:
            print("No entities extracted.")
            return []

        # Split the output into lines
        extracted_titles = ner_output.strip().split('\n')
        # print("DEBUG: Extracted titles from ner_output:", extracted_titles)

        # Filter the extracted titles
        valid_titles = self.filter_titles(extracted_titles, input_text)
        print("DEBUG: Valid entities after filtering:", valid_titles)

        return valid_titles

    def extract_relations(self, input_text):
        """
        Extract entities using the defined pipeline and process the output.

        Args:
            input_text (str): The input text to extract entities from.

        Returns:
            list: Valid extracted entities after filtering.
        """
        # Example structured message to extract movie titles directly
        messages = [
            {"role": "system","content": "No extra text and no answers from you" "You are a question processor, write one word describing the question"
                    "Output only the word user asks about, no titles or names"
                    "E.g. answers: genre, director, publication date, publication date, box office"},
            {"role": "user", "content": "Sentence to extract: " + input_text},
        ]

        # Debugging message to see the exact messages sent to the pipeline
        # print("DEBUG: Messages being sent to pipeline:", messages)

        # Generate the response
        outputs = self.pipe(messages, max_new_tokens=30)

        # Debugging message to see the output from the pipeline
        # print("DEBUG: Outputs received from pipeline:", outputs)

        ner_output = ""

        # Extract the content from the assistant's response
        if isinstance(outputs, list) and len(outputs) > 0 and 'generated_text' in outputs[0]:
            for message in outputs[0]['generated_text']:
                # print("DEBUG: Processing message in output:", message)  # Debugging message
                if isinstance(message, dict) and message.get('role') == 'assistant':
                    ner_output = message.get('content', "")
                    break

        # Debugging message to see what was extracted
        # print("DEBUG: Extracted ner_output:", ner_output)

        # Check if content was found
        if not ner_output:
            print("No entities extracted.")
            return []

        # Split the output into lines
        extracted_titles = ner_output.strip().split('\n')
        # print("DEBUG: Extracted titles from ner_output:", extracted_titles)

        # Filter the extracted titles
        valid_titles = self.filter_titles(extracted_titles, input_text)
        print("DEBUG: Valid relations after filtering:", valid_titles)

        return valid_titles
    
if __name__ == "__main__":
    processor = LlamaProcessor()
    user_input = "Who is the director of Star Wars: Episode VI - Return of the Jedi?"
    extracted_entities = processor.extract_entities(user_input)
    extracted_relations = processor.extract_relations(user_input)

    print(extracted_entities)
    print(extracted_relations)

    if extracted_entities:
        pass
    else:
        print("No valid entities extracted.")