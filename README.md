## Setups

Speakeasy
```bash
pip install speakeasy-python-client-library/dist/speakeasypy-1.0.0-py3-none-any.whl
```

Data
```bash
mkdir dataset data logs
cd dataset
wget -r -np -R "index.html*" https://files.ifi.uzh.ch/ddis/teaching/ATAI2024/dataset/
unzip ddis-movie-graph.nt.zip
mv 14_graph.nt ../data/14_graph.nt
``` 

Run
```bash
python demo_bot.py
```

## Issues
1. https://lms.uzh.ch/auth/RepositoryEntry/17583866519/CourseNode/85421310450657/Message/17608802433 (3rd SPARQL Query Gets Cut Off)

## Project 

### 1st Intermediate Evaluation:  

Answering Simple SPARQL Queries

Description: In the 1st evaluation, your agent needs to be able to answer plain SPARQL queries using the provided knowledge graph. Your agent will receive plain SPARQL queries and respond with the correct answers from the provided knowledge graph.

The objective of this submission is to test if your chatbot can interact with the speakeasy infrastructure, read input SPARQL queries, execute queries over the knowledge graph, and present answers in conversations.

Example queries: 

```
# Which movie has the highest user rating?  

PREFIX ddis: <http://ddis.ch/atai/>   
PREFIX wd: <http://www.wikidata.org/entity/>   
PREFIX wdt: <http://www.wikidata.org/prop/direct/>   
PREFIX schema: <http://schema.org/>   

SELECT ?lbl WHERE {  
    ?movie wdt:P31 wd:Q11424 .  
    ?movie ddis:rating ?rating .  
    ?movie rdfs:label ?lbl .  
}  

ORDER BY DESC(?rating)   
LIMIT 1 
```

['Forrest Gump'] 

 
```
# Which movie has the lowest user rating? 

PREFIX ddis: <http://ddis.ch/atai/>   
PREFIX wd: <http://www.wikidata.org/entity/>   
PREFIX wdt: <http://www.wikidata.org/prop/direct/>   
PREFIX schema: <http://schema.org/>   

SELECT ?lbl WHERE {  
    ?movie wdt:P31 wd:Q11424 .  
    ?movie ddis:rating ?rating .  
    ?movie rdfs:label ?lbl .  
}  

ORDER BY ASC(?rating)   
LIMIT 1 
```

['Vampire Assassin'] 

 
```
# Who directed the movie Apocalypse Now?  

PREFIX ddis: <http://ddis.ch/atai/>   
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>
PREFIX schema: <http://schema.org/>   

SELECT ?director WHERE {  
    ?movie rdfs:label "Apocalypse Now"@en .  
        ?movie wdt:P57 ?directorItem . 
    ?directorItem rdfs:label ?director . 
}  

LIMIT 1  
```

['Francis Ford Coppola'] 

Note: Your bot should expect the plain SPARQL query as input. The natural language questions in the table above are solely to explain the SPARQL queries.

Submission deadline: no report submission

Presentation: no student presentation

Evaluation event date: 14.10.2024 

Requirements: 

Your active participation. 
