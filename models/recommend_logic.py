import pandas as pd
import numpy as np

RECOMMENDATION_TEMPLATE = [
    f"[recommendation] Based on your preferences, I recommend: <recommendations>.",
    f"[recommendation] You might like the following: <recommendations>.",
    f"[recommendation] Here are some suggestions for you: <recommendations>.",
    f"[recommendation] Consider these recommendations: <recommendations>.",
    f"[recommendation] Based on what you like, I recommend: <recommendations>.",
]


def process_recommendation(graph, entities):
    """
    Processes a recommendation request using the detected entities from the query.
    Ensures entities are relevant to movies.
    """
    print(f"Processing recommendation for entities: {entities}")

    # Filter out non-movie-related entities
    movie_entities = [e for e in entities if e['ent_type'] in ['TITLE', 'GENRE', 'DIRECTOR']]
    if not movie_entities:
        return "I couldn't identify any movie-related entities for recommendations. Please provide more context."

    # Fetch recommendations based on the movie entities
    recommendations = recommend_similar_movies(graph, movie_entities)

    if recommendations:
        # Ensure only two recommendations are returned
        recommendations = recommendations[:2]
        recommendations_str = ', '.join(recommendations)
        return np.random.choice(RECOMMENDATION_TEMPLATE).replace("<recommendations>", recommendations_str)
    
    return "I couldn't find any recommendations based on your preferences."


def recommend_similar_movies(graph, entities):
    """
    Recommends similar movies based on the given entities (movies, genres, directors).
    """
    recommendations = []

    for entity in entities:
        # Ensure entity is properly mapped to the graph
        if not entity.get('mapping'):
            continue  # Skip unrecognized entities

        entity_uri = entity['mapping']['uri']
        print(f"Processing entity URI: {entity_uri}")

        # Try to recommend by genre first
        if entity['ent_type'] == 'GENRE':
            genre_recommendations = get_movies_by_genre(graph, entity_uri)
            recommendations.extend(genre_recommendations)

        # Try to recommend by director
        elif entity['ent_type'] == 'DIRECTOR':
            director_recommendations = get_movies_by_director(graph, entity_uri)
            recommendations.extend(director_recommendations)

        # Try to recommend by movie title
        elif entity['ent_type'] == 'TITLE':
            related_recommendations = get_related_movies(graph, entity_uri)
            recommendations.extend(related_recommendations)

    # Remove duplicates and return the results
    return list(set(recommendations))

def get_movies_by_genre(graph, movie_uri):
    """
    Retrieves movies from the same genre as the given movie.
    """
    query = f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT ?similar_movie_label WHERE {{
        <{movie_uri}> wdt:P136 ?genre .
        ?similar_movie wdt:P136 ?genre .
        ?similar_movie rdfs:label ?similar_movie_label .
        FILTER(?similar_movie != <{movie_uri}>)
    }}
    LIMIT 10
    """
    results = graph.query(query)
    return [str(row['similar_movie_label']) for row in results]


def get_movies_by_director(graph, movie_uri):
    """
    Retrieves movies directed by the same director as the given movie.
    """
    query = f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT ?similar_movie_label WHERE {{
        <{movie_uri}> wdt:P57 ?director .
        ?similar_movie wdt:P57 ?director .
        ?similar_movie rdfs:label ?similar_movie_label .
        FILTER(?similar_movie != <{movie_uri}>)
    }}
    LIMIT 10
    """
    results = graph.query(query)
    return [str(row['similar_movie_label']) for row in results]


def get_related_movies(graph, movie_uri):
    """
    Retrieves related movies through general relationships (e.g., sequels, co-actors).
    """
    query = f"""
    PREFIX wdt: <http://www.wikidata.org/prop/direct/>
    SELECT ?related_movie_label WHERE {{
        <{movie_uri}> wdt:P144|wdt:P4969|wdt:P58 ?related_movie .
        ?related_movie rdfs:label ?related_movie_label .
    }}
    LIMIT 10
    """
    results = graph.query(query)
    return [str(row['related_movie_label']) for row in results]
