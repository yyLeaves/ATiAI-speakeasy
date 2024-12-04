import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items # compatibility for pandas 2.*


class EntityProcessor:
    def __init__(self):
        """
        """ 

    def extract_entities(self, text):
        raise NotImplementedError

    

 