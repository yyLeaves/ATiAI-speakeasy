from transformers import BertTokenizer, BertForTokenClassification
import torch

class CustomEntityProcessor:
    def __init__(self, model, tokenizer, device):
        print("Initializing CustomEntityProcessor...")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        print("CustomEntityProcessor initialized.")

    def process(self, text):
        print("Processing input text...")
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,  # Adjust as needed
            add_special_tokens=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = predictions[0].tolist()

        # Debugging output
        print("Tokens:", tokens)
        print("Labels:", [self.model.config.id2label[label] for label in labels])

        entities = self.get_entities(tokens, labels)
        print(f"Entities: {entities}")
        return entities


    def get_entities(self, tokens, labels):
        entities = []
        current_entity = []
        current_label = None
        current_start = None

        for idx, (token, label_id) in enumerate(zip(tokens, labels)):
            label = self.model.config.id2label[label_id]

            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue  # Skip special tokens

            if label.startswith("B-"):
                if current_entity:
                    # Append the current entity before starting a new one
                    entities.append({
                        "entity": self._join_subwords(current_entity),
                        "ent_start": current_start,
                        "ent_end": idx - 1,
                        "ent_type": current_label,
                        "confidence": None  # Add confidence if available
                    })
                current_entity = [token]
                current_label = label[2:]
                current_start = idx

            elif label.startswith("I-") and current_label == label[2:]:
                current_entity.append(token)  # Continue building the current entity

            else:
                if current_entity:
                    entities.append({
                        "entity": self._join_subwords(current_entity),
                        "ent_start": current_start,
                        "ent_end": idx - 1,
                        "ent_type": current_label,
                        "confidence": None
                    })
                current_entity = []
                current_label = None
                current_start = None

        if current_entity:
            entities.append({
                "entity": self._join_subwords(current_entity),
                "ent_start": current_start,
                "ent_end": len(tokens) - 1,
                "ent_type": current_label,
                "confidence": None
            })

        return entities


    def _join_subwords(self, tokens):
        """Helper to join subword tokens correctly."""
        entity = []
        for token in tokens:
            if token.startswith("##"):
                entity[-1] += token[2:]  # Merge subword with previous token
            else:
                entity.append(token)
        return " ".join(entity)


    # def get_entities(self, tokens, labels):
    #     entities = []
    #     current_entity = []
    #     current_label = None
    #     current_start = None

    #     for idx, (token, label_id) in enumerate(zip(tokens, labels)):
    #         label = self.model.config.id2label[label_id]

    #         if token in ["[CLS]", "[SEP]", "[PAD]"]:  # Skip special tokens
    #             continue

    #         if label.startswith("B-"):
    #             # Save the previous entity if it exists
    #             if current_entity:
    #                 entities.append({
    #                     "entity": " ".join(current_entity),
    #                     "ent_start": current_start,
    #                     "ent_end": idx - 1,
    #                     "ent_type": current_label,
    #                     "confidence": None  # Add confidence if available
    #                 })
    #                 current_entity = []

    #             # Start a new entity
    #             current_entity.append(token)
    #             current_label = label[2:]  # Remove the "B-" prefix
    #             current_start = idx

    #         elif label.startswith("I-") and current_label == label[2:]:
    #             # Continue the current entity
    #             current_entity.append(token)

    #         else:
    #             # Save the previous entity and reset
    #             if current_entity:
    #                 entities.append({
    #                     "entity": " ".join(current_entity),
    #                     "ent_start": current_start,
    #                     "ent_end": idx - 1,
    #                     "ent_type": current_label,
    #                     "confidence": None  # Add confidence if available
    #                 })
    #                 current_entity = []
    #                 current_label = None
    #                 current_start = None

    #     # Save the last entity if it exists
    #     if current_entity:
    #         entities.append({
    #             "entity": " ".join(current_entity),
    #             "ent_start": current_start,
    #             "ent_end": len(tokens) - 1,
    #             "ent_type": current_label,
    #             "confidence": None  # Add confidence if available
    #         })

    #     return entities

if __name__ == "__main__":
    # Example input text
    example = "Is Pocahontas, a good movie?"

    # Load the fine-tuned model and tokenizer
    model_path = "ner_models_files/fine_tuned_bert_model"  # Replace with your model path
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForTokenClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.to(device)

    # Initialize the processor with the model, tokenizer, and device
    processor = CustomEntityProcessor(model, tokenizer, device)

    # Process the example text
    entities = processor.process(example)
    # print(entities)



# import torch
# from transformers import BertTokenizer, BertForTokenClassification


# class CustomEntityProcessor:
#     def __init__(self, model_path):
#         print("Initializing CustomEntityProcessor...")
#         # Load your fine-tuned BERT model and tokenizer
#         self.tokenizer = BertTokenizer.from_pretrained(model_path)
#         self.model = BertForTokenClassification.from_pretrained(model_path)
#         self.model.eval()  # Set to evaluation mode
#         print("CustomEntityProcessor initialized.")

#     def process(self, data: str):
#         """
#         Process the input data and extract entities.
#         """
#         self.annotations = self.get_annotations(data)
#         self.entities = self.get_entities(self.annotations)
#         print(f"Entities: {self.entities}")

#     def get_annotations(self, data: str):
#         """
#         Perform token classification on the input data.
#         """
#         # Tokenize input
#         inputs = self.tokenizer(
#             data,
#             return_tensors="pt",
#             truncation=True,
#             max_length=512,
#             add_special_tokens=True,
#             padding="max_length"
#         )

#         # Perform inference
#         with torch.no_grad():
#             outputs = self.model(**inputs)
#         logits = outputs.logits

#         # Get predictions
#         predictions = torch.argmax(logits, dim=2).squeeze().tolist()
#         tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"].squeeze().tolist())

#         # Match tokens and labels
#         entities = []
#         for idx, (token, label_id) in enumerate(zip(tokens, predictions)):
#             if token not in ["[CLS]", "[SEP]", "[PAD]"]:
#                 label = self.model.config.id2label[label_id]
#                 entities.append({
#                     "token": token,
#                     "start": idx,
#                     "end": idx,  # Assuming single token; adjust for subwords if needed
#                     "label": label
#                 })

#         return entities

#     def get_entities(self, annotations):
#         """
#         Convert raw annotations into a structured format.
#         """
#         return [
#             {
#                 "entity": annotation["token"],
#                 "ent_start": annotation["start"],
#                 "ent_end": annotation["end"],
#                 "ent_type": annotation["label"]
#             }
#             for annotation in annotations
#         ]


# if __name__ == "__main__":
#     # Example input
#     example = "I want to watch the movie Harry Potter and the Chamber of Secrets."

#     # Provide the path to your trained model
#     model_path = "ner_models_files/fine_tuned_bert_model"

#     # Initialize the processor
#     processor = CustomEntityProcessor(model_path=model_path)

#     # Process the input text
#     processor.process(example)

#     # Print the structured entities
#     print(processor.entities)
