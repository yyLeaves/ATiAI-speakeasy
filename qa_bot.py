from speakeasypy import Speakeasy, Chatroom
from typing import List
import time, os
import configparser
from rdflib import Graph
from utils import logger


config = configparser.ConfigParser()
config.read('config.ini')

username = config['credentials']['username']
password = config['credentials']['password']

bot_name = config['credentials']['bot_name']
bot_pass = config['credentials']['bot_pass']

DEFAULT_HOST_URL = 'https://speakeasy.ifi.uzh.ch'

listen_freq = 2


class Agent:
    def __init__(self, username, password):

        logger.info("Initializing knowledge graph...")
        graph = Graph().parse('data/14_graph.nt', format='turtle')
        logger.info("Knowledge graph initialized.")
        from models.graph import QueryEngine
        self.query_engine = QueryEngine(graph=graph)
        logger.info("Agent initialized.")
        self.username = username
        self.speakeasy = Speakeasy(host=DEFAULT_HOST_URL, username=username, password=password)
        self.speakeasy.login()  # This framework will help you log out automatically when the program terminates.
        logger.info("Agent logged in.")

    def listen(self):
        while True:
            # only check active chatrooms (i.e., remaining_time > 0) if active=True.
            rooms: List[Chatroom] = self.speakeasy.get_rooms(active=True)
            for room in rooms:
                if not room.initiated:
                    # send a welcome message if room is not initiated
                    # room.post_messages(f'Hello from {os.getcwd()}') # for testing
                    logger.info(f"Chatroom {room.room_id} initiated from {os.getcwd()}.")
                    room.post_messages(f'Hello! | Tag! | Ciao! | Grüezi! | Zao! | Annannyeong! | Ohayo! From {room.my_alias}~'
                                       f'You can ask your questions her. I\'m only able to process one simple question at a time.')
                    room.initiated = True
                # Retrieve messages from this chat room.
                # If only_partner=True, it filters out messages sent by the current bot.
                # If only_new=True, it filters out messages that have already been marked as processed.
                for message in room.get_messages(only_partner=True, only_new=True):
                    logger.info(
                        f"\t- Chatroom {room.room_id} "
                        f"- new message #{message.ordinal}: '{message.message}' "
                        f"- {self.get_time()}")

                    # Implement your agent here #
                    try:
                        query_respond = self.query_engine.answer(message.message)
                        logger.info(f"Query result: {query_respond}")
                        room.post_messages(f"{query_respond}")
                    except Exception as e:
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
