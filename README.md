## Setups

Speakeasy
```bash
pip install speakeasy-python-client-library/dist/speakeasypy-1.0.0-py3-none-any.whl
```

Data
```bash
mkdir dataset, data, logs
cd dataset
wget -r -np -R "index.html*" https://files.ifi.uzh.ch/ddis/teaching/ATAI2024/dataset/
unzip ddis-movie-graph.nt.zip
mv 14_graph.nt ../data/14_graph.nt
``` 

**1st Intermediate Evaluation**

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


### 2nd Intermediate Evaluation:  
Description:  In the 2nd evaluation, your agent needs to be able to answer factual questions and embedding questions. You will receive real-world natural language questions. Your agent needs to interpret the given questions, transform them into executable SPARQL queries (just as one possible solution), and fetch the correct answers from the provided knowledge graph.

Example queries/questions: 
```
Who is the director of Star Wars: Episode VI - Return of the Jedi? 

I think it is Richard Marquand. 
```
```
Who is the screenwriter of The Masked Gang: Cyprus? 

The answer suggested by embeddings: Cengiz Küçükayvaz. 
```
```
When was "The Godfather" released? 

It was released in 1972. 
```

Please note that you may consider a given question as

1) a factual question by providing the factual answer from the knowledge graph,

2) an embedding question by providing the embedding computation-based answer,

3) or both by providing both answers.

To make sure that other people can give accurate ratings and not misunderstand your embedding answer as a factual answer (or vice versa), please clearly state what kind of answer is given in your response, e.g., the second question above. Also, we recommend using "(Embedding Answer)" to mark each embedding answer.

Report Submission deadline: 28.10.2024 (23:59 Zurich time)

Student Presentation: Multiple teams will be selected to give a 10-minute presentation (including QA). We will inform the selected teams 2 days before the event.

Evaluation event date: 04.11.2024 

Requirements: 

For the 2nd intermediate evaluation, you need to: 

Submit the latest version of your code in a compressed file. The name of the file should be: ATAI_Eval2_[bot_name].zip, where [bot_name] is the name of your team's bot.
Submit a report using the provided template (c.f. Project - Overview).
Actively participate in the event.  