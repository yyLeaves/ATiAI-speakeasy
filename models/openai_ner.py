from openai import OpenAI
import os
import json

class OpenAIEntityProcessor:
    def __init__(self, api_key=None):
        print("Initializing OpenAIEntityProcessor...")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        print("OpenAIEntityProcessor initialized.")

    def process(self, text):
        print("Processing input text with OpenAI...")
        try:
            # Use GPT-4 for better contextual understanding
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert entity extraction tool. Extract movie-related entities such as "
                            "TITLE, ACTOR, DIRECTOR, and GENRE. Output the entities as a JSON array of objects with "
                            "The movies and actors may be not well knows as well..... you have to search very deep! Do internet search if not sure!"
                            "keys: 'entity', 'ent_type'."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Extract entities from the following text: {text}",
                    },
                ],
                temperature=0.0  # To ensure deterministic responses
            )
            return self.parse_response(response)
        except Exception as e:
            print(f"Error processing input text: {e}")
            return []

    def parse_response(self, response):
        """
        Parse the response from the OpenAI API.
        Assumes the assistant returns a JSON-like structure or a readable list of entities.
        """
        try:
            content = response.choices[0].message.content
            print(f"Raw response: {content}")

            # Expecting a valid JSON response
            entities = json.loads(content)  # Convert JSON string to Python list/dict
            return entities
        except json.JSONDecodeError as e:
            print(f"Error parsing response as JSON: {e}")
            return []

if __name__ == "__main__":
    example_text = "Let me know what Sandra Bullock looks like."
    
    # Replace with your OpenAI API key
    api_key = "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"

    processor = OpenAIEntityProcessor(api_key=api_key)
    entities = processor.process(example_text)
    print("Entities:", entities)

    # api_key = "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"