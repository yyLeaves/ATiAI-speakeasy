from speakeasypy import Speakeasy, Chatroom
from typing import List
import time, os, random
import configparser
from rdflib import Graph
from utils import logger
from models.embedding import GraphEmbedding
from models.recommender import RecommendEngine
from models.intention import IntentionDetection
from models.postprocess import MovieNameProcessor, EntityNameProcessor
from models.graph import QueryEngine
from models.crowdsourcing import CrowdsourcingProcessor
from models.merge_answers import ResponseMerger
from models.multimedia import MultimediaModel
from models.relation import RelationProcessor, RelationMapper
from models.llama.llama_detector import LlamaProcessor  # Import your LlamaEntityProcessor here
# from models.llama.relations_llama_movie import LlamaRelationProcessor  # Import your LlamaEntityProcessor here

import torch
from transformers import pipeline

config = configparser.ConfigParser()
config.read('config.ini')

username = config['credentials']['username']
password = config['credentials']['password']

bot_name = config['credentials']['bot_name']
bot_pass = config['credentials']['bot_pass']

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'

listen_freq = 2

# Global Torch settings for deterministic behavior
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)
torch.manual_seed(42)

# Check for MPS availability and set the device accordingly
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Using device: {device}")

# # Path to your downloaded Llama model
# model_id = "./Llama-3.2-3B-Instruct"

class Agent:
    def __init__(self, username, password):

        logger.info("Initializing knowledge graph...")
        graph = Graph().parse('data/14_graph.nt', format='turtle')
        self.graph = graph
        logger.info("Knowledge graph initialized.")

        self.int_det = IntentionDetection()

        # Initialize the LlamaEntityProcessor using the preloaded pipeline
        self.llama_processor = LlamaProcessor()  # Make sure to pass the preloaded pipeline
        # self.llama_relation_processor = LlamaRelationProcessor(pipe)  # Make sure to pass the preloaded pipeline
        self.mnp = MovieNameProcessor()
        self.enp = EntityNameProcessor()

        self.recommender = RecommendEngine(graph=self.graph)
        self.mmm = MultimediaModel()
        
        self.query_engine = QueryEngine(graph=graph)
        self.embedding = GraphEmbedding(graph=graph)
        self.crowdsourcing = CrowdsourcingProcessor()
        self.responsemerger = ResponseMerger()

        logger.info("Agent initialized.")
        
        self.username = username
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.
        logger.info("Agent logged in.")
        self.history = {}

    def listen(self):
        while True:
            # only check active chatrooms (i.e., remaining_time > 0) if active=True.
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    # send a welcome message if room is not initiated
                    logger.info(f"Chatroom {room.room_id} initiated from {os.getcwd()}.")
                    room.post_messages(f'Salut! '
                                       f"I'm your personal chatbot. How can I help you today? One query at a time please.")
                    room.initiated = True

                for message in room.get_messages(only_partner=True, only_new=True):
                    logger.info(
                        f"\t- Chatroom {room.room_id} "
                        f"- new message #{message.ordinal}: '{message.message}' "
                        f"- {self.get_time()}")
                    try:
                        if message.message.strip() in self.history:
                            response = self.history[message.message.strip()]
                            logger.info(f"Query result: {response}")
                            room.post_messages(f"{response}.")
                            room.mark_as_processed(message)
                            continue

                        else:
                            intention = self.int_det.detect_intention(message.message)
                            if intention == 'recommend':
                                logger.info(f"Detected intention: {intention}")

                                # movie_names = self.query_engine.get_entities(message.message) # list of movie names extracted using NER
                                # movie_matches = self.mnp.process(movie_names)
                                # movie_matches = [movie for movie in movie_matches if movie['mapping'] is not False]
                                # logger.info(f"Extracted Movie Names: {movie_names}")
                                
                                # Use preloaded LlamaEntityProcessor to extract movie names
                                movie_names = self.llama_processor.extract_entities(message.message)
                                movie_matches = self.mnp.process(movie_names)
                                movie_matches = [movie for movie in movie_matches if movie['mapping'] is not False]
                                logger.info(f"Extracted Movie Names: {movie_names}")
                                
                                if len(movie_matches) == 0:
                                    # Actor name, director name, genre, etc.
                                    alternative_entities = self.llama_processor.extract_entities(message.message) # NEW implemented
                                    ent_matches = self.enp.process(alternative_entities)
                                    ent_matches = [entity for entity in ent_matches if entity['mapping'] is not False]
                                    # TODO: if movies is empty, search for other entities and recommend based on it
                                    logger.info(f"Entity Matches: {ent_matches}")
                                    feature_rec = self.recommender.recommend_by_entity(ent_matches)
                                    logger.info(f"Feature Recommendation: {feature_rec}")
                                    response = ','.join(feature_rec)

                                # elif len(movie_matches) == 1: # TODO: strategy for single movie
                                #     # recommendation based on genre, ...
                                #     ...
                                else:
                                    movie_matches = self.mnp.process(movie_names)
                                    movie_matches = [movie for movie in movie_matches if movie['mapping'] is not False]

                                    logger.info(f"Movie Matches: {movie_matches}")
                                    feature_rec = self.recommender.recommend_by_movie_feature(movie_matches)
                                    logger.info(f"Feature Recommendation: {feature_rec}")

                                    knn_rec = self.recommender.recommend_by_knn(movie_matches)
                                    logger.info(f"KNN Recommendation: {knn_rec}")

                                    intersection = set(feature_rec).intersection(set(knn_rec))
                                    rec_movies = []
                                    if len(intersection) > 0:
                                        rec_movies += list(intersection)[:3]
                                    rec_movies += knn_rec[:5] + feature_rec[:5]
                                    rec_movies = list(set(rec_movies))
                                    random.shuffle(rec_movies)
                                    print(f"Recommendation: {rec_movies}")
                                    rec_movie_names = [self.recommender.id2movie[movie] for movie in rec_movies]
                                    logger.info(f"Recommendation: {rec_movie_names}")

                                    response = self.recommender.prepare_response(rec_movie_names)
                                
                                print(f"Response: {response}")


                            elif intention == 'multimedia':
                                logger.info(f"Detected intention: {intention}")
                                ent_names = self.llama_processor.extract_entities(message.message) # TODO: extract movie/actor name
                                ent_matches = self.enp.process(ent_names)
                                ent_matches = [entity for entity in ent_matches if entity['mapping'] is not False]
                                logger.info(f"Entity Matches: {ent_names, ent_matches}")

                                r = self.mmm.get_imbd_ids(ent_matches)
                                imdb_person = r['person']
                                imdb_movie = r['movie']
                                logger.info(f"IMDB IDs: {imdb_person, imdb_movie}")
                                image = self.mmm.get_image(imdb_person, imdb_movie)
                                response = self.mmm.prepare_response(imdb_person, imdb_movie, image)
                                
                            else:
                                # TODO: implement 
                                response = None
                                query = message.message
                                entities = self.llama_processor.extract_entities(message.message) # TODO: implement it anywhere
                                entities = self.mnp.process(entities)
                                logger.info(f"Entities: {entities}")
                                # rp = RelationProcessor(message.message, entities)
                                relation_mapper = RelationMapper()
                                # logger.info(f"RP_new: {rp}")
                                relation_llama = self.llama_processor.extract_relations(message.message)
                                # logger.info(f"RP_new: {rp1}")
                                # relation, pid = rp.parse()
                                if relation_llama:
                                    relation, pid = relation_mapper.mapping(relation_llama[0])
                                else:
                                    print("No valid relation extracted")
                                    relation, pid = None, None
                                # logger.info(f"Relation: {relation}, PID: {pid}")
                                if (relation is not None) and (pid is not None):
                                    # print("HERE ARE ENTITIES:")
                                    # print(entities)
                                    results = self.query_engine.query(query, entities, relation, pid)
                                    # logger.info(f"Results:, {results}")
                                    graph_results = [self.query_engine.translate_res(res) for res in results]
                                    # logger.info(f"Results 1 2 3:, {results}")
                                    graph_response = self.query_engine._format_answer(graph_results)
                                    print("GRAPH ANSWER:", graph_response)
                                    # crowd_response = self.crowdsourcing.generate_answer(graph_results, entities, pid)
                                    crowd_response = self.crowdsourcing.generate_answer(graph_results, entities, pid)
                                    print("CROWD ANSWER:", crowd_response)


                                    response = self.responsemerger.merge_responses(graph_response, crowd_response)

                                    # logger.info(f"{response}")
                                    # if response is None:
                                    #     print("We did not manage to find the answer directly in the graph. Let's see what crowdsourcing says!")

                                if response is None:
                                    if len(entities) == 0 and pid is None:
                                        response = "I'm sorry, I couldn't find the answer to your question. Can you please rephrase it?"
                                    else:
                                        logger.info("Trying to answer with embeddings.")
                                        top_match = self.embedding.retrieve(entities, relation)
                                        print(f"Top match: {top_match}")
                                    if top_match:
                                        if len(top_match) > 1:
                                            results = top_match[0]
                                        else:
                                            results = top_match
                                    results = [self.query_engine.translate_res(res) for res in results]
                                    response = self.query_engine._format_answer(results, embedding=True)

                                    print("Embeddings...")
                                """Try to answer in the order of
                                1. graph search ->
                                2. crowdsource ->
                                3. embeddings ->
                                """
                                if response is None:
                                    response = "I'm sorry, I couldn't find the answer to your question. Can you please rephrase it?"

                            # response = f"Movies: {movies}"
                            logger.info(f"Query result: {response}")
                            self.history[message.message.strip()] = response
                            room.post_messages(f"{response}.")
                            room.mark_as_processed(message)
                    except Exception as e:
                        print(f"Error processing query: {e}")
                        logger.error(f"Error processing query: {e}")
                        room.post_messages(f"Sorry, your query could not be processed. Please input the correct query.")
                    # Mark the message as processed, so it will be filtered out when retrieving new messages.
                    room.mark_as_processed(message)

                # Retrieve reactions from this chat room.
                # If only_new=True, it filters out reactions that have already been marked as processed.
                for reaction in room.get_reactions(only_new=True):
                    logger.info(
                        f"\t- Chatroom {room.room_id} "
                        f"- new reaction #{reaction.message_ordinal}: '{reaction.type}' "
                        f"- {self.get_time()}")

                    # Implement your agent here #
                    room.post_messages(f"Received your reaction: '{reaction.type}' ")
                    room.mark_as_processed(reaction)

            time.sleep(listen_freq)

    @staticmethod
    def get_time():
        return time.strftime("%H:%M:%S, %d-%m-%Y", time.localtime())


if __name__ == '__main__':
    demo_bot = Agent(bot_name, bot_pass)
    demo_bot.listen()
