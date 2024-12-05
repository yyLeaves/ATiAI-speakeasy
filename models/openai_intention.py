from openai import OpenAI
import os

class RecommendationIntentDetector:
    def __init__(self, api_key=None):
        print("Initializing RecommendationIntentDetector...")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        print("RecommendationIntentDetector initialized.")

    def is_recommendation(self, query):
        """
        Determines if the user query is seeking a recommendation.
        :param query: User's input string
        :return: 1 if it's a recommendation query, 0 otherwise
        """
        try:
            print("Analyzing query for recommendation intent...")
            response = self.client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an intent detection system. Determine if the given query is asking for a "
                            "recommendation. Respond with '1' if the user is looking for a recommendation and '0' otherwise."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Is the following query asking for a recommendation? Query: {query}",
                    },
                ],
                temperature=0.0  # Deterministic response
            )
            return self.parse_response(response)
        except Exception as e:
            print(f"Error analyzing intent: {e}")
            return 0

    def parse_response(self, response):
        """
        Parse the response from OpenAI to extract the intent.
        :param response: The raw response from OpenAI
        :return: 1 or 0 based on intent
        """
        try:
            content = response.choices[0].message.content.strip()
            print(f"Raw response: {content}")
            return int(content)
        except ValueError as e:
            print(f"Error parsing response as integer: {e}")
            return 0

if __name__ == "__main__":
    example_query = "Who is the director of Titanic?"
    
    # Replace with your OpenAI API key
    api_key = "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"

    detector = RecommendationIntentDetector(api_key=api_key)
    result = detector.is_recommendation(example_query)
    print("Is recommendation:", result)  # Expected: 1

# api_key = "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"
