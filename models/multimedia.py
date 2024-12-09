import pandas as pd
import numpy as np

class MultimediaModel:

    def __init__(self):
        """"""
        self.df_imdb = pd.read_pickle("data/df_spo_imdb.pkl")
        self.imdb2id = dict(zip(self.df_imdb['o'], self.df_imdb['sid']))
        self.df_images = pd.read_pickle("data/df_images.pkl")
        self.df_person = pd.read_pickle("data/df_person.pkl")
        self.df_person['id'] = self.df_person['uri'].apply(lambda x: x.split('/')[-1].strip('>'))
        self.person2movie = dict(zip(self.df_person['id'], self.df_person['label']))
        self.list_person = self.df_person['id'].values
        self.df_movie = pd.read_pickle("data/df_movie.pkl")
        self.df_movie['id'] = self.df_movie['uri'].apply(lambda x: x.split('/')[-1].strip('>'))
        self.id2movie = dict(zip(self.df_movie['id'], self.df_movie['label']))
        self.list_movie = self.df_movie['id'].values


    # def extract_name(self, query):
    #     return ["The Matrix", "Keanu Reeves"] # TODO: implement this anywhere else
    
    def get_imbd_ids(self, entity_ids):
        ids = [ent['mapping']['uri'].split('/')[-1] for ent in entity_ids]
        imdb_persons = []
        imdb_movie = None
        for ent_id in ids:
            if ent_id in self.list_person:
                imdb_persons.append(self.df_imdb[self.df_imdb['sid']==ent_id].iloc[0]['o'])
            elif ent_id in self.list_movie:
                imdb_movie = self.df_imdb[self.df_imdb['sid']==ent_id].iloc[0]['o']
        return {
            'person': imdb_persons,
            'movie': imdb_movie
        }
    
    def prepare_response(self, imdb_persons, imdb_movie, image):

        templates_no_movie = (
            "Sure, here is an image of 【person】 for you.",
            "Here is the image of 【person】 you wanted.",
            "Let me show you the image of 【person】.",
            "Of course, here is the image of 【person】.",
            "Why not? Here is the image of 【person】.",
        )
        
        templates_movie = (
            "Alright, this is the image of 【person】 in 【movie】.",
            "This is the image of 【person】 in 【movie】.",
            "Let me show you the image of 【person】 in 【movie】.",
            "Here is the image of 【person】 in 【movie】 that you asked for.",
            "Of course, here is the image of 【person】 in 【movie】.",
        )

        if image is None:
            return "Sorry, I couldn't find any image for the given query"
        
        if imdb_movie is not None:
            imdb_movie = self.imdb2id[imdb_movie]
            imdb_movie = self.id2movie[imdb_movie]
            imdb_persons = [self.imdb2id[person] for person in imdb_persons]
            imdb_persons = [self.person2movie[person] for person in imdb_persons]            
            template = np.random.choice(list(templates_movie))
            res = template.replace("【person】", ', '.join(imdb_persons))
            res = res.replace("【movie】", imdb_movie)
        else:
            imdb_persons = [self.imdb2id[person] for person in imdb_persons]
            imdb_persons = [self.person2movie[person] for person in imdb_persons]
            template = np.random.choice(list(templates_no_movie))
            res = template.replace("【person】", ', '.join(imdb_persons))
        return f"{res}\n{image}"


    
    def get_image(self, imdb_persons, imdb_movie):
        df_images = self.df_images
        if imdb_movie is not None:
            df_images = df_images[df_images.apply(lambda x: imdb_movie in x['movie'], axis=1)]
        for p in imdb_persons:
            df_images = df_images[df_images.apply(lambda x: p in x['cast'], axis=1)]
        if len(df_images) > 0:
            if 'poster' in df_images['type']:
                df_images = df_images[df_images['type']=='poster']
            image = df_images.loc[df_images['h'].idxmax(), 'img']
            image = f"image:{image.split('.')[0]}"
            return image
        return None