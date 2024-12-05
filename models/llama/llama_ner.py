import os
import json
from transformers import LlamaForCausalLM, LlamaTokenizer

class LlamaEntityProcessor:
    def __init__(self, model, tokenizer):
        print("Initializing LlamaEntityProcessor...")
        self.tokenizer = tokenizer
        self.model = model
        self.model.eval()  # Set to evaluation mode
        print("LlamaEntityProcessor initialized.")

    def process(self, text):
        print("Processing input text with Llama...")
        try:
            input_text = (
                "You are an expert entity extraction tool. Extract movie-related entities such as "
                "TITLE, ACTOR, DIRECTOR, and GENRE. Output the entities as a JSON array of objects. "
                "Keys: 'entity', 'ent_type'."
                "\n\n" + text
            )
            inputs = self.tokenizer(input_text, return_tensors="pt")
            outputs = self.model.generate(**inputs, max_length=512, num_return_sequences=1, do_sample=False)

            # Decode the generated text
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            return self.parse_response(generated_text)
        except Exception as e:
            print(f"Error processing input text: {e}")
            return []

    def parse_response(self, response):
        """
        Parse the response from Llama model.
        Expects the model to output JSON-like text with entities.
        """
        try:
            print(f"Raw response: {response}")
            
            # Expecting a JSON-like structure
            entities = json.loads(response)  # Convert JSON string to Python list/dict
            return entities
        except json.JSONDecodeError as e:
            print(f"Error parsing response as JSON: {e}")
            return []

if __name__ == "__main__":
    example_text = "Can you tell me the publication date of Tom Meets Zizou?"

    model_path = "/Users/dawid/.llama/checkpoints/Llama3.2-3B-Instruct"
    
    # Load model and tokenizer using transformers
    model = LlamaForCausalLM.from_pretrained(model_path)
    tokenizer = LlamaTokenizer.from_pretrained(model_path)
    
    # Initialize the processor with the loaded model and tokenizer
    processor = LlamaEntityProcessor(model=model, tokenizer=tokenizer)
    entities = processor.process(example_text)
    print("Entities:", entities)