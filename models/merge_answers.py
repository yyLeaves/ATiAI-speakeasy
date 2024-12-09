import pandas as pd
import numpy as np
import random

class ResponseMerger:
    def __init__(self):
        self.no_graph_template = [
            "[graph] Our graph method could not help with this question but let's see what the crowdsourcing says.",
            "[graph] Unfortunately, our database does not have an answer to this query.",
            "[graph] The graph data is silent on this one.",
            "[graph] No graph-based answers available for this.",
            "[graph] The graph is unable to provide any insights into this question."
        ]

        self.no_crowd_template = [
            "[crowd] The crowdsourcing data does not include any cues for that question.",
            "[crowd] No crowd-based insights available for this question.",
            "[crowd] Our crowd sources were unable to provide an answer here.",
            "[crowd] Crowdsourcing turned up empty this time.",
            "[crowd] The crowd could not contribute any information for this query."
        ]

    def merge_responses(self, graph_response, crowd_response):
        if not graph_response and not crowd_response:
            return None

        if graph_response and not crowd_response:
            # Select a random template stating the crowd's response is missing
            no_crowd_msg = random.choice(self.no_crowd_template)
            return f"{graph_response}\n{no_crowd_msg}"
        
        if crowd_response and not graph_response:
            # Select a random template stating the graph's response is missing
            no_graph_msg = random.choice(self.no_graph_template)
            return f"{no_graph_msg}\n{crowd_response}"
        
        if graph_response and crowd_response:
            # If the responses are the same, just return one; otherwise, return both.
            if graph_response == crowd_response:
                return graph_response
            return f"{graph_response}\n{crowd_response}"

# Example usage
if __name__ == "__main__":
    merger = ResponseMerger()
    graph_resp = "Our graph analysis shows it's related to environmental factors."
    crowd_resp = None
    merged_response = merger.merge_responses(graph_resp, crowd_resp)
    print(merged_response)
