from rapidfuzz import fuzz
import os
import pandas as pd
from models.data_config import (
    FUZZ_THRESHOLD, 
    FUZZ_THRESHOLD_LOW,
    INVALID_ENTITY,
    MOVIE_NER_TYPES,
    DATA_PATH)

class PostProcessor:

    def __init__(self, text, movie_entities, all_entities):
        self.text = text
        self.movie_entities = movie_entities
        self.all_entities = all_entities
        self._read_data()
        print("PostProcessor initialized.")

    def process(self) -> list:
        print("Processing entities...")
        self.mapping()
        self.merge_entities()
        print("Entities processed.")
        return self.merged_entities
        # return {
        #     "movie_entities": self.movie_entities,
        #     "all_entities": self.all_entities,
        #     "merged_entities": self.merged_entities
        # }


    def mapping(self):
        # mapping movie entities
        # REVIEW | AWARD | DIRECTOR | RATING | \RATINGS_AVERAGE\
        # GENRE | CHARACTER | SONG | ACTOR | TITLE | RELATIONSHIP | YEAR
        for entity in self.movie_entities:
            # TODO: could be one to many mapping
            search_res = self._entity_search(entity['entity']) 
            if search_res:
                entity['mapping'] = search_res
            else:
                match_res = self._entity_match(entity['entity'])
                if match_res:
                    entity['mapping'] = match_res
                else:
                    entity['mapping'] = False

        # mapping all other entities
        for entity in self.all_entities:
            # exact match
            search_res = self._entity_search(entity['entity'])
            if search_res:
                entity['mapping'] = search_res
            # fuzzy match
            else:
                # print(entity)
                match_res = self._entity_match(entity['entity'])
                if match_res:
                    entity['mapping'] = match_res
                # no match
                else:
                    entity['mapping'] = False

    def merge_entities(self):
        merged_entities = []
        text = self.text
        for movie_ent in self.movie_entities:
            if self.is_valid_movie_entity(movie_ent, text):
                # get rid of duplicate entities
                for ent in self.all_entities:
                    if (ent['ent_start'] == movie_ent['ent_start'] and 
                        ent['ent_end'] >= movie_ent['ent_end'] and 
                        ent['confidence'] >= movie_ent['confidence']):
                        
                        merged_entities.append(ent)
                        text = text[:ent['ent_start']] + (ent['ent_end']-ent['ent_end']+1)*'Æ' + text[ent['ent_end']:]
                        break
                else:
                    # No token entity found, add the movie entity
                    merged_entities.append(movie_ent)
                    text = text[:movie_ent['ent_start']] + (movie_ent['ent_end']-movie_ent['ent_start']+1)*'Æ' + text[movie_ent['ent_end']:]

        for ent in self.all_entities: 
            if text[ent['ent_start']:ent['ent_end']+1] == ent['entity']:
                merged_entities.append(ent)
        self.merged_entities = merged_entities

    def _read_data(self):
        self.df_uri = pd.read_pickle(os.path.join(DATA_PATH, 'df_uri.pkl'))
        self.df_ent = pd.read_pickle(os.path.join(DATA_PATH, 'df_ent.pkl'))
        self.df_rel = pd.read_pickle(os.path.join(DATA_PATH, 'df_rel.pkl'))
        self.df_genre = pd.read_pickle(os.path.join(DATA_PATH, 'df_genre.pkl'))
        self.df_person = pd.read_pickle(os.path.join(DATA_PATH, 'df_person.pkl'))
        self.df_movie = pd.read_pickle(os.path.join(DATA_PATH, 'df_movie.pkl'))

    def _entity_search(self, entity):
        res = self.df_ent[self.df_ent['label']==entity]
        if len(res) > 0:
            search = res.iloc[0].to_dict() # TODO: return first match for now
            return search
        else:
            return False
      
    def _entity_match(self, entity):
        """fuzzy matching of entity"""
        match_scores = self.df_ent['label'].apply(lambda x: fuzz.ratio(x.lower(), entity.lower()))
        match_score = match_scores.max()

        if match_score > FUZZ_THRESHOLD_LOW:
            match_idx = match_scores.idxmax()
            match = self.df_ent.iloc[match_idx].to_dict()
            match['match_score'] = match_score
            return match
        return False

    # [{'ent_start': 26, 'ent_end': 64, 'enntity': 'Harry Potter and the Chamber of Secrets', 'ent_type': 'TITLE', 'confidence': '0.8125714'}]    
    def is_valid_movie_entity(self, entity, text_left):
        return entity['ent_type'] in MOVIE_NER_TYPES and entity['entity'] not in INVALID_ENTITY and entity['entity'] in text_left