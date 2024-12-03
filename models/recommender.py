from sklearn.neighbors import NearestNeighbors

from models.embedding import GraphEmbedding
from collections import Counter
import numpy as np
import pandas as pd

class RecommendEngine():

    def __init__(self, graph):
        self.embedding_model = GraphEmbedding(graph=graph)
        self.graph = graph
        self.df_movie_spo = pd.read_pickle("D:/Project/ATiAI-speakeasy/data/df_movie_spo.pkl")
        self.df_movie_emb = pd.read_pickle("D:/Project/ATiAI-speakeasy/data/df_movie_emb.pkl")
        df_movie = pd.read_pickle("D:/Project/ATiAI-speakeasy/data/df_movie.pkl")
        df_movie['id'] = df_movie['uri'].apply(lambda x: x.split('/')[-1])
        self.id2movie = {row['id']: str(row['label']) for idx, row in df_movie.iterrows()}

    def recommend_by_movie_feature(self, movies, topk=50):
        movie_ids = [str(movie['mapping']['uri']).split('/')[-1] for movie in movies if movie['mapping'] is not False]
        movie_names = [self.id2movie[movie] for movie in movie_ids] 
        pos = self.get_common_po(movie_ids)
        if pos is None:
            return None
        common_movies = self.find_common_movies(pos, movie_ids, topk=topk)

        recommend_movies = [m[0] for m in common_movies]
        print(f"Recommend Movies: {[self.id2movie[m] for m in recommend_movies]}")
        return recommend_movies 

    def recommend_by_knn(self, movies, topk=30):
        movie_ids = [str(movie['mapping']['uri']).split('/')[-1] for movie in movies]
        movie_names = [self.id2movie[movie] for movie in movie_ids]

        X = self.df_movie_emb['emb'].tolist()
        nbrs = NearestNeighbors(n_neighbors=topk*len(movie_ids)*2, algorithm='auto').fit(X)
        reference_points = [self.df_movie_emb[self.df_movie_emb['id']==movie]['emb'].item().tolist() for movie in movie_ids]
        rec_ids = []
        for p in reference_points:
            _, indices = nbrs.kneighbors([p])
            rec_ids += indices[0].tolist()
        distances, indices = nbrs.kneighbors(reference_points)
        list_of_movies = self.df_movie_emb['id'].values 
        rec_ids = [list_of_movies[i] for i in indices[0]]
        rec_ids = [rec for rec in rec_ids if rec not in movies]
        top_recs = Counter(rec_ids).most_common(topk)
        top_recs = [rec[0] for rec in top_recs]

        top_recs = [rec for rec in top_recs if rec not in movie_ids]
        top_recs = [rec for rec in top_recs if self.id2movie[rec] not in movie_names]
        print("KNN Recommendation: ", [self.id2movie[m] for m in top_recs])
        return top_recs

    def recommend_by_entity(self, entities):
        raise NotImplementedError

    def prepare_response(self, movie_names):
        response = ["Here are some movies you might like: \n", "Based on your preferences, I recommend the following movies: \n", "The following movies are recommended based on your preferences: \n", "Why not try these movies: \n"]
        movies = ', '.join(movie_names)
        response = np.random.choice(response) + movies
        return response

    def recommend(self):
        pos = self.get_common_po(self.movies)
        common_movies = self.find_common_movies(pos)
        print(f"Common Movies: {common_movies}")
        recommend_movies = [m[0] for m in common_movies][:10]
        print(f"Recommend Movies: {recommend_movies}")
        movie_names = [self.id2movie[movie] for movie in recommend_movies]
        movie_names = [movie for movie in movie_names if movie not in self.movie_names]
        print(f"Movie Names: {movie_names}")
        return self.prepare_response(movie_names)


    def get_common_po(self, movies, topk=15):
        pos = []
        for movie in movies:
            print(f"Movie: {movie}")
            po = self.df_movie_spo[self.df_movie_spo['sid']==movie]
            if len(po) == 0:
                continue
            pos += po['po'].values[0]
        po_counts = Counter(pos)
        common_counts = po_counts.most_common(topk)
        print(f"Top Common Features: {common_counts}")
        if len(common_counts) == 0:
            return None
        
        if common_counts[0][0] == len(movies):
            count_num = max(1, len(movies)-1)
        else:
            count_num = 1

        return [po for po, count in common_counts if count >= count_num]

        
    def find_common_movies(self, pos, movie_ids, topk=20):
        list_movies = []
        for p in pos:
            df_p = self.df_movie_spo[self.df_movie_spo.apply(lambda x: p in x['po'], axis=1)]
            list_movies += df_p['sid'].values.tolist()
        list_movies = [movie for movie in list_movies if movie not in movie_ids]
        movie_names = [self.id2movie[movie] for movie in movie_ids]
        list_movies = [movie for movie in list_movies if self.id2movie[movie] not in movie_names] # avoid same name different id
        most_common = Counter(list_movies).most_common(topk)
        print(f"Most Common: {most_common}")
        return most_common
    
    def recommend_embedding(self, movie_ids, topk=30):
        list_of_movies = []
        list_of_embeddings = []
        for movie in self.movie_uris:
            embedding = self.embedding_model.get_entity_embedding_single(movie)
            # print(f"{movie} Embedding: {embedding}")
            if movie is None:
                continue
            most_similar = self.embedding_model.get_most_similar(embedding, topn=topk)
            print(f"Most Similar: {most_similar}")
            most_similar = [movie['Entity'] for movie in most_similar if movie['Entity'] not in self.movie_names]
            # print(f"Most Similar: {most_similar}")
            list_of_movies+=most_similar
            # print(f"Most Similar: {list_of_movies}")
            list_of_embeddings.append(embedding)
            # print(f"List of Embedding: {list_of_embeddings}")
        movie_counts = Counter(list_of_movies)
        # print(f"Movie Counts: {movie_counts}")
        average_embedding = self.embedding_model.get_average_embedding(list_of_embeddings)
        # print(f"Average Embedding: {average_embedding}")
        average_similar = self.embedding_model.get_most_similar(average_embedding, topn=topk)
        # print(f"Average Similar: {average_similar}")
        average_similar = [movie['Entity'] for movie in average_similar if movie['Entity'] not in self.movies]
        list_of_movies += average_similar
        # print(f"List of Movies: {list_of_movies}")
        movie_counts = Counter(list_of_movies).most_common(topk)
        # print(f"Movie Counts: {movie_counts}")
        # return movie has the highest count
        # common_counts = movie_counts.most_common(topk)
        # print(f"Common Counts: {common_counts}")
        # return common_counts
        return movie_counts
        # return [c[0] for c in common_counts]
    
    from collections import Counter

    def combine_most_common(self, c1, c2):
        d1 = {c[0]: c[1] for c in c1}
        d2 = {c[0]: c[1] for c in c2}
        dicts = [d1, d2]
        all_keys = set().union(*dicts)
        print(f"All Keys: {all_keys}")
        combined = {k: sum(d.get(k, 0) for d in dicts) for k in all_keys}
        print(f"Combined: {sorted(combined.items(), key=lambda x: x[1], reverse=True)}")
        return sorted(combined.items(), key=lambda x: x[1], reverse=True)
        