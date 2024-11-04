import re
import numpy as np
import pandas as pd
from rdflib import Namespace, Graph, RDFS

import sys
sys.path.append('D:/Project/ATiAI-speakeasy/')

from models.entity import EntityProcessor, MovieEntityProcessor
from models.postprocess import PostProcessor
from models.relation import RelationProcessor
from models.data_config import QUERY_PREFIX
from models.embedding import GraphEmbedding



ANSWER_TEMPLATE = [
    f"[graph] From my knowledge, the answer is <answer>.",
    f"[graph] I think the answer to your question is <answer>.",
    f"[graph] I believe the answer is <answer>.",
    f"[graph] Just checked my database, the answer should be <answer>.",
    f"[graph] My knowledge database tells me the answer is <answer>.",
]

ANSWER_TEMPLATE_EMBEDDING = [
    f"[embedding] The answer could be <answer>.",
    f"[embedding] I think the answer might be <answer>.",
    f"[embedding] My embedding suggests <answer>.",
    f"[embedding] I hope my embeddings won't fail me, so I would say the answer could be <answer>.",
    f"[embedding] I guess the answer could be <answer>!",
]


class QueryEngine():

    def __init__(self, graph):
        self.graph = graph
        # self.graph.parse('D:/Project/ATiAI-speakeasy/data/14_graph.nt', format='turtle')
        self.embedding = GraphEmbedding(graph=self.graph)
        self.ep = EntityProcessor()
        self.mp = MovieEntityProcessor()
        print("Query Engine initialized.")

    def answer(self, query: str):
        self.query = query
        print(f"Trying to answer: {query}")
        entities = self.get_entities(query)
        print(f"Gathered entities: {entities}")
        relation, pid = self.get_relations(entities)
        print(f"Gathered relation: {relation}-{pid}")   
        results = self._query(query, entities, relation, pid)
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
                    # results = ', '.join(top_match[0])
                    if len(top_match) > 1:
                        # results = ', '.join(top_match)
                        results = top_match[0]
                    else:
                        results = top_match
                    return np.random.choice(list(ANSWER_TEMPLATE_EMBEDDING)).replace("<answer>", results)
        return "I'm sorry, I couldn't find the answer to your question. Can you plase rephrase it?"
        # ANSWER_TEMPLATE

    def translate_res(self, res):
        print(f"Translating: {res}")
        if "entity/Q" in res:
            ent_id = res.split('/')[-1].strip('>')
            df = pd.read_pickle("D:/Project/ATiAI-speakeasy/data/df_ent.pkl")
            lbl = df[df['uri'].str.contains(ent_id)].iloc[0]['label']
            print(f"Translated: {lbl}")
            return lbl
        return res

    def _query(self, query: str, entities, relation, pid):
        print("Querying process started.")
        if len(entities) == 0 or pid is None:
            return None
        
        ent_labels = [e['mapping']['label'] for e in entities if e['mapping'] is not False]
        ent_ids = [str(e['mapping']['uri']).split('/')[-1] for e in entities if e['mapping'] is not False]

        ent_id = ent_ids[0]
        df = pd.read_csv('D:/Project/ATiAI-speakeasy/dataset/14_graph.tsv', sep='\t', header=None)
        res = df[df[1].str.contains(pid) & df[0].str.contains(ent_id)][2].to_list()

        print(f"Results: {res}")
        return res

        # print(f"query e{entities} r{relation}-{pid}")
        # entity_labels = ' '.join(f'"{label}"@en' for label in ent_labels)

        # if pid == "P577":
        #     query = f'''{QUERY_PREFIX}
        #     SELECT ?releaseDate 
        #     WHERE {{
        #         VALUES ?movieLabel {{ {entity_labels} }}
        #         ?movie rdfs:label ?movieLabel .
        #         ?movie wdt:P31 wd:Q11424 .
        #         ?movie wdt:P577 ?releaseDate .
        #     }}
        #     '''

        # else:
        #     query = f'''{QUERY_PREFIX}
        #     SELECT ?lbl WHERE {{
        #     VALUES ?movieLabel {{ {entity_labels} }}
        #     ?movie rdfs:label ?movieLabel .
        #     ?movie wdt:{pid} ?{relation} .
        #     ?{relation} rdfs:label ?lbl .
        #     }}'''

        # print(f"Graph query sent: {query}")

        # # TODO: add restrictions
        # # restrictions = get_restrictions(query)
        # # if restrictions['flag']:
        # #     RESTRICTIONS = f"ORDER BY {restrictions['order']}(?rating) LIMIT {restrictions['number']}"
        # #     query = f"""{query}
        # #             {RESTRICTIONS}"""
        # try:
        #     print(f"Trying to query: {query}")
        #     results = [str(r) for r, in self.graph.query(query)]
        # except Exception as exception:
        #     print(f"Factual Error: {type(exception).__name__}")
        #     return None
        # return results
    
    def _format_answer(self, answer):
        answer = ', '.join(answer)
        # answer = pd.DataFrame(answer,columns=answer.vars)
        # lens = answer.shape[0]
        # items = answer.iloc[:,0].tolist()
        # if lens > 1:
        #     last_item = items.pop()
        #     answer = ', '.join(items) + ' and ' + last_item
        # elif items:
        #         answer = items[0]
        # else:
        #     answer = None
        return answer


    def get_entities(self, query: str):
        self.ep.process(query)
        self.mp.process(query)
        movie_entities = self.mp.entities
        all_entities = self.ep.entities
        print(f"get movie entities: {movie_entities} and all entities: {all_entities}")
        pp = PostProcessor(query, movie_entities, all_entities)
        mapping_entities = pp.process()

        return mapping_entities

    def get_relations(self, entities):
        rp = RelationProcessor(self.query, entities)
        relation, pid = rp.parse()
        return relation, pid
    

def get_restrictions(q: str):
    DESC = False
    AESC = False
    desc_pattern = r"(?:top|best|highest|most|good|lowest|worse|bad|bottom|least)?\s*(\d+|one|two|three|four|five)\s*(?:good|best|highest|most|top|lowest|worse|bad|bottom|least)?"
    matches = re.findall(desc_pattern, q, re.IGNORECASE)
    flag = False
    for match in matches:
        number = matches[0] if matches[0] else 3 # TODO: default number
        if number in ["one", "two", "three", "four", "five"]:
            number = ["one", "two", "three", "four", "five"].index(number) + 1
        if re.findall(r"top|best|highest|most|good", q, re.IGNORECASE):
            order = "DESC"
            flag = True
        elif re.findall(r"lowest|worse|bad|bottom|least", q, re.IGNORECASE):
            order = "ASC"
            flag = True
        else:
            order = None
    return {
        "flag": flag,
        "number": number,
        "order": order
    }

