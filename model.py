import rdflib
graph = rdflib.Graph()
graph.parse('data/14_graph.nt', format='turtle')

def graph_query(query_string:str):
    return [str(s) for s, in graph.query(query_string)]