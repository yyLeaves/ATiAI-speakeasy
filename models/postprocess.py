from rapidfuzz import fuzz
import os
import pandas as pd
from models.data_config import (
    FUZZ_THRESHOLD, 
    FUZZ_THRESHOLD_LOW,
    INVALID_ENTITY,
    MOVIE_NER_TYPES,
    DATA_PATH)


class MovieNameProcessor:
    """Responsible for mapping movie names from the input text to its graph mapping."""
    def __init__(self):
        self.df_movie = pd.read_pickle(os.path.join(DATA_PATH, 'df_movie.pkl'))

    def process(self, movie_names: list):
        """ [{'entity': 'Harry Potter and the Chamber of Secrets', 'mapping': {'uri': rdflib.term.URIRef('http://www.wikidata.org/entity/Q1148981'), 'label': 'Harry Potter and the Chamber of Secrets'}, 'confidence': 1}, {'entity': 'Harry Potter and the Goblet of Fire', 'mapping': {'uri': rdflib.term.URIRef('http://www.wikidata.org/entity/Q849901'), 'label': 'Harry Potter and the Goblet of Fire'}, 'confidence': 1}, {'entity': 'Harry Potter and the Order of the Phoenix', 'mapping': {'uri': rdflib.term.URIRef('http://www.wikidata.org/entity/Q1148993'), 'label': 'Harry Potter and the Order of the Phoenix'}, 'confidence': 1}]"""
        print("Processing movie names...")
        self.name_list = [{'entity': name} for name in movie_names]
        for i, name_dict in enumerate(self.name_list):
            name = name_dict['entity']
            search_res = self._entity_search(name) # exact search
            if search_res:
                self.name_list[i]['mapping'] = search_res
                self.name_list[i]['confidence'] = 1
            else:
                match_res = self._entity_match(name) # fuzzy match
                if match_res:
                    self.name_list[i]['mapping'] = match_res
                    self.name_list[i]['confidence'] = match_res['match_score']
                else:
                    self.name_list[i]['mapping'] = False
        return self.name_list

    def _entity_search(self, entity):
        """exact search of entity"""
        res = self.df_movie[self.df_movie['label'].str.lower()==entity.lower()]
        fuz_res = self.df_movie[self.df_movie['label']==f"the {entity.lower()}"] # deal with 'the' in movie names
        if len(res) > 0:
            search = res.iloc[0].to_dict() # TODO: return first match for now
            return search
        elif len(fuz_res) > 0:
            search = fuz_res.iloc[0].to_dict()
            return search
        else:
            return False

    def _entity_match(self, entity):
        """fuzzy matching of entity"""
        df_candidates = self.df_movie[self.df_movie['label'].str.contains(entity.lower())]
        if len(df_candidates) > 0:
            match_scores = df_candidates['label'].apply(lambda x: fuzz.ratio(x.lower(), entity.lower()))
            match_score = match_scores.max()
            if match_score > FUZZ_THRESHOLD_LOW-0.1:
                match_idx = match_scores.idxmax()
                match = self.df_movie.iloc[match_idx].to_dict()
                match['match_score'] = match_score
                return match

        else:
            match_scores = self.df_movie['label'].apply(lambda x: fuzz.ratio(x.lower(), entity.lower()))
            match_score = match_scores.max()

        if match_score > FUZZ_THRESHOLD_LOW:
            match_idx = match_scores.idxmax()
            match = self.df_movie.iloc[match_idx].to_dict()
            match['match_score'] = match_score
            return match
        return False
    

class EntityNameProcessor:
    """Responsible for mapping movie names from the input text to its graph mapping."""
    def __init__(self):
        self.df_ent = pd.read_pickle(os.path.join(DATA_PATH, 'df_ent.pkl'))

    def process(self, ent_names: list):
        """ [{'entity': 'Harry Potter and the Chamber of Secrets', 'mapping': {'uri': rdflib.term.URIRef('http://www.wikidata.org/entity/Q1148981'), 'label': 'Harry Potter and the Chamber of Secrets'}, 'confidence': 1}, {'entity': 'Harry Potter and the Goblet of Fire', 'mapping': {'uri': rdflib.term.URIRef('http://www.wikidata.org/entity/Q849901'), 'label': 'Harry Potter and the Goblet of Fire'}, 'confidence': 1}, {'entity': 'Harry Potter and the Order of the Phoenix', 'mapping': {'uri': rdflib.term.URIRef('http://www.wikidata.org/entity/Q1148993'), 'label': 'Harry Potter and the Order of the Phoenix'}, 'confidence': 1}]"""
        print("Processing movie names...")
        self.name_list = [{'entity': name} for name in ent_names]
        for i, name_dict in enumerate(self.name_list):
            name = name_dict['entity']
            search_res = self._entity_search(name) # exact search
            if search_res:
                self.name_list[i]['mapping'] = search_res
                self.name_list[i]['confidence'] = 1
            else:
                match_res = self._entity_match(name) # fuzzy match
                if match_res:
                    self.name_list[i]['mapping'] = match_res
                    self.name_list[i]['confidence'] = match_res['match_score']
                else:
                    self.name_list[i]['mapping'] = False
        return self.name_list

    def _entity_search(self, entity):
        """exact search of entity"""
        res = self.df_ent[self.df_ent['label'].str.lower()==entity.lower()]
        fuz_res = self.df_ent[self.df_ent['label']==f"the {entity.lower()}"] # deal with 'the' in movie names
        if len(res) > 0:
            search = res.iloc[0].to_dict() # TODO: return first match for now
            return search
        elif len(fuz_res) > 0:
            search = fuz_res.iloc[0].to_dict()
            return search
        else:
            return False

    def _entity_match(self, entity):
        """fuzzy matching of entity"""
        df_candidates = self.df_ent[self.df_ent['label'].str.contains(entity.lower())]
        if len(df_candidates) > 0:
            match_scores = df_candidates['label'].apply(lambda x: fuzz.ratio(x.lower(), entity.lower()))
            match_score = match_scores.max()
            if match_score > FUZZ_THRESHOLD_LOW-0.1:
                match_idx = match_scores.idxmax()
                match = self.df_ent.iloc[match_idx].to_dict()
                match['match_score'] = match_score
                return match

        else:
            match_scores = self.df_ent['label'].apply(lambda x: fuzz.ratio(x.lower(), entity.lower()))
            match_score = match_scores.max()

        if match_score > FUZZ_THRESHOLD_LOW:
            match_idx = match_scores.idxmax()
            match = self.df_ent.iloc[match_idx].to_dict()
            match['match_score'] = match_score
            return match
        return False


if __name__ == "__main__":
    
    movie_names = ['Harry Potter and the Chamber of Secrets', 'Harry Potter and the Goblet of Fire', 'Harry Potter and the Order of the Phoenix']
    processor = MovieNameProcessor()
    res = processor.process(movie_names)
    print(res)
    # [{'entity': 'Harry Potter and the Chamber of Secrets', 'mapping': {'uri': rdflib.term.URIRef('http://www.wikidata.org/entity/Q1148981'), 'label': 'Harry Potter and the Chamber of Secrets'}, 'confidence': 1}, {'entity': 'Harry Potter and the Goblet of Fire', 'mapping': {'uri': rdflib.term.URIRef('http://www.wikidata.org/entity/Q849901'), 'label': 'Harry Potter and the Goblet of Fire'}, 'confidence': 1}, {'entity': 'Harry Potter and the Order of the Phoenix', 'mapping': {'uri': rdflib.term.URIRef('http://www.wikidata.org/entity/Q1148993'), 'label': 'Harry Potter and the Order of the Phoenix'}, 'confidence': 1}]