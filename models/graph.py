import re
import numpy as np
import pandas as pd
from rdflib import Namespace, Graph, RDFS

import sys
sys.path.append('')

from models.postprocess import MovieNameProcessor
from models.relation import RelationProcessor
from models.data_config import QUERY_PREFIX
from models.embedding import GraphEmbedding


ANSWER_TEMPLATE = [
    f"[graph] From my knowledge, the answer is <answer>",
    f"[graph] I think the answer to your question is <answer>",
    f"[graph] I believe the answer is <answer>",
    f"[graph] Just checked my database, the answer should be <answer>",
    f"[graph] My knowledge database tells me the answer is <answer>",
]

ANSWER_TEMPLATE_EMBEDDING = [
    f"[embedding] The answer could be <answer>",
    f"[embedding] I think the answer might be <answer>",
    f"[embedding] My embedding suggests <answer>",
    f"[embedding] I hope my embeddings won't fail me, so I would say the answer could be <answer>",
    f"[embedding] I guess the answer could be <answer>!",
]

class QueryEngine():

    def __init__(self, graph, entity_labels_path = "data/df_ent.pkl"):
        self.graph = graph
        self.embedding = GraphEmbedding(graph=self.graph)
        self.entity_labels = pd.read_pickle(entity_labels_path)

    def answer_graph(self, query: str):
        entities = self.get_list_movies(query)

        relation, pid = self.get_relations(query, entities)
        print(f"Gathered relation: {relation}-{pid}")

        results = self.query(query, entities, relation, pid)
        
        results = [self.translate_res(res) for res in results]
        print(f"Results: {results}")
        if results and len(results) > 0:
            answer = self._format_answer(results)
            if answer is not None:
                return np.random.choice(list(ANSWER_TEMPLATE)).replace("<answer>", answer)
        else:
            # embedding
            if len(entities) == 0 and pid is None:
                return "I'm sorry, I couldn't find the answer to your question. Can you please rephrase it?"
            else:
                print("Trying to answer with embeddings.")
                top_match = self.embedding.retrieve(entities, relation)
                print(f"Top match: {top_match}")
                if top_match:
                    if len(top_match) > 1:
                        results = top_match[0]
                    else:
                        results = top_match
                    return np.random.choice(list(ANSWER_TEMPLATE_EMBEDDING)).replace("<answer>", results)
        return "I'm sorry, I couldn't find the answer to your question. Can you plase rephrase it?"
        # ANSWER_TEMPLATE

    # def translate_res(self, res):
    #     print(f"Translating: {res}")
    #     if "entity/Q" in res:
    #         ent_id = res.split('/')[-1].strip('>')
    #         df = pd.read_pickle("data/df_ent.pkl")
    #         lbl = df[df['uri'].str.contains(ent_id)].iloc[0]['label']
    #         print(f"Translated: {lbl}")
    #         return lbl
    #     return res

    def translate_res(self, res):
        print(f"Checking: {res}")
        # Check if the response is a Wikidata URI or not
        if "wikidata.org/entity/Q" in res:
            # Use regex to find an exact Wikidata entity ID pattern
            match = re.search(r'Q\d+', res)
            if match:
                ent_id = match.group(0)  # Get the full entity ID
                # Filter the DataFrame for an exact match in the 'uri' column
                label_row = self.entity_labels[self.entity_labels['uri'].str.endswith(ent_id)]

                if not label_row.empty:
                    lbl = label_row.iloc[0]['label']  # Get the label corresponding to the entity ID
                    print(f"Translated: {lbl}")
                    return lbl
                else:
                    return "UNKNOWN"  # Return 'UNKNOWN' if no matching ID found
        # If the response does not need translation or is a direct answer
        return res
    
    def query(self, query: str, entities, relation, pid):
        print("Querying process started.")
        if len(entities) == 0 or pid is None:
            return None
        
        ent_ids = [str(e['mapping']['uri']).split('/')[-1] for e in entities if e['mapping'] is not False]

        ent_id = ent_ids[0]
        df = pd.read_csv('dataset/14_graph.tsv', sep='\t', header=None)
        res = df[df[1].str.contains(pid) & df[0].str.contains(ent_id)][2].to_list()

        print(f"Results: {res}")
        return res

    def _format_answer(self, answer: list, embedding=False):
        if not embedding:
            if answer is not None and len(answer) > 0:
                template = np.random.choice(list(ANSWER_TEMPLATE)).replace("<answer>", ', '.join(answer))
                return template
            return None
        else:
            if answer is not None and len(answer) > 0:
                template = np.random.choice(list(ANSWER_TEMPLATE_EMBEDDING)).replace("<answer>", ', '.join(answer))
                return template
        return None


    # def get_list_movies(self, query: str):
    #     # TODO: change interface here
    #     print("List of movie names")
    #     raise NotImplementedError
    #     return movies

    def get_relations(self, query, entities):
        rp = RelationProcessor(query, entities)
        relation, pid = rp.parse()
        return relation, pid
    

