INVALID_ENTITY = ['MPAA', 'PG', 'PG-13', 'R', 'G', 'NC-17', 'NR', 'UA']
MOVIE_NER_TYPES = [
    # 'REVIEW', 
                   'AWARD', 'DIRECTOR', 'RATING', 'RATINGS_AVERAGE', 'GENRE',
                   'CHARACTER', 'SONG', 'ACTOR', 'TITLE', 'YEAR', 'RELATIONSHIP']
FUZZ_THRESHOLD = 95
FUZZ_THRESHOLD_LOW = 80
DATA_PATH = "data"

PAT_EXTRACT = r"\b(What|When|Where|Why|How|How many|Tell me|Tell me about|Show me|Show me about|show|give|Give me|Let me know|Do you know|Who|Which|is|are|was|were|will|would|could|can|should|does|do|did|the|a|an|in|on|at|for|with|by|to|'s|'re|be|being|been)\b|\.|\?"

EMBEDDING_MODEL = "data/models/all-MiniLM-L12-v2"

from rdflib import Namespace
WD = Namespace('http://www.wikidata.org/entity/')
WDT = Namespace('http://www.wikidata.org/prop/direct/')
SCHEMA = Namespace('http://schema.org/')
DDIS = Namespace('http://ddis.ch/atai/')
QUERY_PREFIX = \
"""PREFIX ddis: <http://ddis.ch/atai/> 
PREFIX wd: <http://www.wikidata.org/entity/> 
PREFIX wdt: <http://www.wikidata.org/prop/direct/> 
PREFIX schema: <http://schema.org/>
"""
