from transformers import BertTokenizer, BertForTokenClassification
import torch
import pandas as pd

pd.DataFrame.iteritems = pd.DataFrame.items  # Compatibility for pandas 2.x


class EntityProcessor:
    """
    Uses the pre-trained BERT NER model provided in the original pipeline.
    """
    def __init__(self, model, tokenizer, device):
        print("Initializing EntityProcessor...")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        print("EntityProcessor initialized.")

    def process(self, text):
        print("Processing input text...")
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
            add_special_tokens=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = predictions[0].tolist()

        print("Tokens:", tokens)
        print("Labels:", [self.model.config.id2label[label] for label in labels])

        entities = self.get_entities(tokens, labels)
        print(f"General Entities: {entities}")
        return entities

    def get_entities(self, tokens, labels):
        entities = []
        current_entity = []
        current_label = None
        current_start = None

        for idx, (token, label_id) in enumerate(zip(tokens, labels)):
            label = self.model.config.id2label[label_id]

            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            if label.startswith("B-"):
                if current_entity:
                    entities.append({
                        "entity": " ".join(current_entity),
                        "ent_start": current_start,
                        "ent_end": idx - 1,
                        "ent_type": current_label,
                        "confidence": None
                    })
                current_entity = [token]
                current_label = label[2:]
                current_start = idx

            elif label.startswith("I-") and current_label == label[2:]:
                current_entity.append(token)

            else:
                if current_entity:
                    entities.append({
                        "entity": " ".join(current_entity),
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
                "entity": " ".join(current_entity),
                "ent_start": current_start,
                "ent_end": len(tokens) - 1,
                "ent_type": current_label,
                "confidence": None
            })

        return entities


class MovieEntityProcessor:
    """
    Uses the fine-tuned BERT model for movie-specific entities.
    """
    def __init__(self, model, tokenizer, device):
        print("Initializing MovieEntityProcessor...")
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()
        print("MovieEntityProcessor initialized.")

    def process(self, text):
        print("Processing input text for movie-specific entities...")
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=128,
            add_special_tokens=True
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        predictions = torch.argmax(outputs.logits, dim=2)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        labels = predictions[0].tolist()

        print("Tokens:", tokens)
        print("Labels:", [self.model.config.id2label[label] for label in labels])

        entities = self.get_entities(tokens, labels)
        print(f"Movie Entities: {entities}")
        return entities

    def get_entities(self, tokens, labels):
        entities = []
        current_entity = []
        current_label = None
        current_start = None

        for idx, (token, label_id) in enumerate(zip(tokens, labels)):
            label = self.model.config.id2label[label_id]

            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                continue

            if label.startswith("B-"):
                if current_entity:
                    entities.append({
                        "entity": " ".join(current_entity),
                        "ent_start": current_start,
                        "ent_end": idx - 1,
                        "ent_type": current_label,
                        "confidence": None
                    })
                current_entity = [token]
                current_label = label[2:]
                current_start = idx

            elif label.startswith("I-") and current_label == label[2:]:
                current_entity.append(token)

            else:
                if current_entity:
                    entities.append({
                        "entity": " ".join(current_entity),
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
                "entity": " ".join(current_entity),
                "ent_start": current_start,
                "ent_end": len(tokens) - 1,
                "ent_type": current_label,
                "confidence": None
            })

        return entities


if __name__ == "__main__":
    example = "Given that I like The Lion King and Pocahontas, can you recommend some movies?"

    # Load the fine-tuned model and tokenizer for movie-specific entities
    movie_model_path = "ner_models_files/fine_tuned_bert_model"  # Replace with the fine-tuned model path
    movie_tokenizer = BertTokenizer.from_pretrained(movie_model_path)
    movie_model = BertForTokenClassification.from_pretrained(movie_model_path)

    # Load the pre-trained BERT model and tokenizer for general entities
    general_model_path = "bert-base-cased"  # Replace with the original BERT model
    general_tokenizer = BertTokenizer.from_pretrained(general_model_path)
    general_model = BertForTokenClassification.from_pretrained(general_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    movie_model.to(device)
    general_model.to(device)

    # Initialize processors
    movie_processor = MovieEntityProcessor(movie_model, movie_tokenizer, device)
    general_processor = EntityProcessor(general_model, general_tokenizer, device)

    # Process example text
    movie_entities = movie_processor.process(example)
    general_entities = general_processor.process(example)

    print("Movie Entities:", movie_entities)
    print("General Entities:", general_entities)
