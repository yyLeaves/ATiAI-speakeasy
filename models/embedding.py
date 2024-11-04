import numpy as np
import pandas as pd
import rdflib
from rdflib import Graph, Namespace, RDFS
import csv

WD = Namespace('http://www.wikidata.org/entity/')
WDT = Namespace('http://www.wikidata.org/prop/direct/')
DDIS = Namespace('http://ddis.ch/atai/')
RDFS = rdflib.namespace.RDFS
SCHEMA = Namespace('http://schema.org/')
from sklearn.metrics import pairwise_distances


class GraphEmbedding:
    
    def __init__(self, graph):
        self.graph = graph
        # self.graph.parse('data/14_graph.nt', format='turtle')
        self.ent2lbl = {ent: str(lbl) for ent, lbl in self.graph.subject_objects(RDFS.label)}
        self.lbl2ent = {lbl: ent for ent, lbl in self.ent2lbl.items()}

        self.entity_emb = np.load('D:/Project/ATiAI-speakeasy/data/ddis-graph-embeddings/entity_embeds.npy')
        self.relation_emb = np.load('D:/Project/ATiAI-speakeasy/data/ddis-graph-embeddings/relation_embeds.npy')

        with open('D:/Project/ATiAI-speakeasy/data/ddis-graph-embeddings/entity_ids.del', 'r') as ifile:
            self.ent2id = {rdflib.term.URIRef(ent): int(idx) for idx, ent in csv.reader(ifile, delimiter='\t')}
            self.id2ent = {v: k for k, v in self.ent2id.items()}
        with open('D:/Project/ATiAI-speakeasy/data/ddis-graph-embeddings/relation_ids.del', 'r') as ifile:
            self.rel2id = {rdflib.term.URIRef(rel): int(idx) for idx, rel in csv.reader(ifile, delimiter='\t')}
            self.id2rel = {v: k for k, v in self.rel2id.items()}

        df_rel_extended = pd.read_pickle("D:/Project/ATiAI-speakeasy/data/df_rel_extend.pkl")
        # self.lbl2rel = {row['relation']: row['relation_label'] for idx, row in df_rel_extended.iterrows()}
        # self.rel2lbl = {row['relation_label']: row['relation'] for idx, row in df_rel_extended.iterrows()}

        self.lbl2rel = {row['label']: row['id'] for idx, row in df_rel_extended.iterrows()}
        self.rel2lbl = {row['id']: row['label'] for idx, row in df_rel_extended.iterrows()}
        # self.movie_ent2lbl = {}
        # self.movie_ent2id = {}

        print("Graph Embedding initialized.")

    def retrieve(self, entities, relation):
        print(f"Retrieving entities: {entities} and relation: {relation}")
        ent_emb, rel_emb = self.get_embedding(entities, relation)
        top_match = self.match_most_likely(ent_emb, rel_emb, 3)
        return top_match

    def get_embedding(self, entities, relation):
        entities = [ent for ent in entities if ent['mapping'] is not False]
        for ent in entities:
            label = ent['mapping']['label']
            print(f"Entity label: {label}")
            try:
                # ent_embs = [self.entity_emb[self.ent2id[self.lbl2ent[label]]] for ent in entities if ent in self.lbl2ent.keys()]
                ent_embs = [self.entity_emb[self.ent2id[self.lbl2ent[label]]] for ent in entities if ent['mapping']['label'] in self.lbl2ent.keys()]
            except KeyError:
                ent_embs = np.zeros(256)
        ent_emb = np.mean(ent_embs, axis=0) if len(ent_embs) > 0 else np.zeros(256)
        print(f"len(ent_embs): {len(ent_embs)}")
        rel_emb = self.relation_emb[self.rel2id[WDT[self.lbl2rel[relation]]]] if relation in self.lbl2rel.keys() else np.zeros(256)
        print("Got embeddings.")
        return ent_emb, rel_emb
 
    def match_most_likely(self, ent_emb, rel_emb, topn):
        lhs = np.zeros(256)
        lhs = np.add(lhs, ent_emb)
        lhs = np.add(lhs, rel_emb)
        if np.all(lhs == 0):
            return None
        distances = pairwise_distances(lhs.reshape(1, -1), self.entity_emb).reshape(-1).argsort()
        most_likely_results_df = pd.DataFrame([
            (self.id2ent[idx][len(WD):], self.ent2lbl[self.id2ent[idx]], distances[idx], rank+1)
            for rank, idx in enumerate(distances[:topn])],
            columns=('Entity', 'Label', 'Score', 'Rank'))
        most_likely_results = most_likely_results_df.to_dict('records')
        print(f"Most likely results: {most_likely_results}")
        return [result['Label'] for result in most_likely_results]


if __name__ == "__main__":
    ...