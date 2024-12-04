import re
import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items # compatibility for pandas 2.*

from typing import Any, List

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import torch.nn.functional as F

from sklearn.metrics.pairwise import cosine_similarity

from .data_config import EMBEDDING_MODEL

from .data_config import PAT_EXTRACT


class Text2VecEmbedding():

    def __init__(self, model_name_or_path, device='cpu') -> None:
        self.device = device
        self.tokenizer = BertTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
        self.model = BertModel.from_pretrained(model_name_or_path, trust_remote_code=True).to(self.device)

    def __call__(self, source) -> List:
        return self.embed(source)

    def embed(self, source) -> List:
        if isinstance(source, str):
            return self.get_single_embedding(source)
        elif isinstance(source, list):
            return self.get_batch_embedding(source)

    def get_single_embedding(self, text):
        return self.get_batch_embedding([text])[0]

    def get_batch_embedding(self, texts) -> List:
        inputs = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt',
                                max_length=256).to(self.device)
        with torch.no_grad():
            model_output = self.model(**inputs)
            embeddings = self.mean_pooling(model_output, inputs['attention_mask'])
            embeddings = self.normalize(embeddings)
            embeddings = self.cpu(embeddings)
            embeddings = self.tolist(embeddings)
        return embeddings

    def normalize(self, embedding):
        return F.normalize(embedding)

    def cpu(self, embeddings):
        return embeddings.cpu()

    def tolist(self, embeddings):
        return embeddings.tolist()

    # Mean Pooling - Take attention mask into account for correct averaging
    @staticmethod
    def mean_pooling(model_output, attention_mask):
        token_embeddings = model_output[0]
        # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) \
            / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


class RelationProcessor:

    def __init__(self, query, entities):
        self.query = query
        # self.list_entities = [e['entity'] for e in entities]
        self.df_rel_extended_embed = pd.read_pickle("D:/Project/ATiAI-speakeasy/data/df_rel_extend_embed.pkl")

    def parse(self):
        relation = self.extract_relation(self.query)
        relation, pid = self.mapping(relation)
        return relation, pid # check not None

    def extract_relation(self, q):
        cleaned_text = re.compile(r'\b(?:' + '|'.join(map(re.escape, self.query)) + r')\b', re.IGNORECASE).sub("", q)
        for pat, rep in [(PAT_EXTRACT, "")]:
            cleaned_text = re.sub(pat, rep, cleaned_text, flags=re.IGNORECASE)
            relation = cleaned_text.strip().replace('"', '')
        if relation == "":
            if "when" in q:
                relation = "release date"
            elif "MPAA" in q:
                relation = "MPAA film rating"
            elif "rating" in q or "rate" in q:
                relation = "MPAA film rating"
            else:
                # TODO: 
                relation = None
        return relation
    
    def _cosine_similarity(self, a, b):
        dot_product = np.dot(a, b)
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        return dot_product / (norm_a * norm_b)
    
    def mapping(self, relation):
        # extract match
        if relation is None:
            return None
        res = self.df_rel_extended_embed[self.df_rel_extended_embed['label']==relation]
        if len(res) > 0:
            id = res.iloc[0]['id']
            label = relation
        else: # similarity
            embedding_model = Text2VecEmbedding(EMBEDDING_MODEL, device='cpu')
            relation_embedding = embedding_model.embed(relation)
            similarity = self.df_rel_extended_embed['embedding'].apply(lambda x: self._cosine_similarity(x, relation_embedding))
            idx = similarity.idxmax()
            id = self.df_rel_extended_embed.iloc[idx]['id']
            label = self.df_rel_extended_embed.iloc[idx]['label']
        return label, id
            

def main():

    rp = RelationProcessor("Who directs Harry Potter?", [{'entity': 'Harry Potter', 'ent_start': 21, 'ent_end': 32, 'ent_type': 'TITLE', 'confidence': '0.96424997', 'mapping': {'uri': 'http://www.wikidata.org/entity/Q3244512', 'label': 'Harry Potter'}}])
    rp.mapping("directs")
    print(rp.pid)


if __name__ == "__main__":
    main()
