import pandas as pd
pd.DataFrame.iteritems = pd.DataFrame.items # compatibility for pandas 2.*

from sparknlp.base import *
from sparknlp.annotator import *
import sparknlp
from pyspark.ml import Pipeline
from sparknlp.annotator import *
from sparknlp.base import *

spark = sparknlp.start()


# https://sparknlp.org/2021/08/05/bert_base_token_classifier_conll03_en.html
class EntityProcessor:

    def __init__(self,):
        print("Initializing EntityProcessor...")
        self.pipeline = self.get_pipeline()
        print("EntityProcessor initialized.")

    def process(self, data: str):
        self.annotations = self.get_annotations(data)
        self.entities = self.get_entities(self.annotations)
        print(f"get general entities: {self.entities}")

    def get_annotations(self, data: str):
        df = spark.createDataFrame(pd.DataFrame({'text': [data]}))
        res = self.pipeline.fit(df).transform(df)
        return res.select("entities").collect()[0][0]

    def get_entities(self, data: list):
        list_entities = [self._create_entity_dict(row) 
                         for row in data]
        return list_entities 

    def _create_entity_dict(self, row):
        print(row)
        return {            
            "entity": row.result,
            "ent_start": row.begin,
            "ent_end": row.end,
            "ent_type": row.metadata['entity'],
            "confidence": row.metadata['confidence']
            }

    def get_pipeline(self):
        document_assembler = DocumentAssembler() \
        .setInputCol('text') \
        .setOutputCol('document')

        tokenizer = Tokenizer() \
        .setInputCols(['document']) \
        .setOutputCol('token')

        tokenClassifier = BertForTokenClassification \
        .pretrained('bert_base_token_classifier_conll03', 'en') \
        .setInputCols(['token', 'document']) \
        .setOutputCol('ner') \
        .setCaseSensitive(True) \
        .setMaxSentenceLength(512)

        # since output column is IOB/IOB2 style, NerConverter can extract entities
        ner_converter = NerConverter() \
        .setInputCols(['document', 'token', 'ner']) \
        .setOutputCol('entities')

        pipeline = Pipeline(stages=[
        document_assembler, 
        tokenizer,
        tokenClassifier,
        ner_converter
        ])

        return pipeline
    

# https://sparknlp.org/2021/07/20/ner_mit_movie_simple_distilbert_base_cased_en.html
class MovieEntityProcessor:

    def __init__(self,):
        print("Initializing MovieEntityProcessor...")
        self.pipeline = self.get_pipeline()
        print("MovieEntityProcessor initialized.")

    def process(self, data: str):
        self.annotations = self.get_annotations(data)
        self.entities = self.get_entities(self.annotations)
        print(f"get movie entities: {self.entities}")

    def get_annotations(self, data: str):
        df = spark.createDataFrame(pd.DataFrame({'text': [data]}))
        res = self.pipeline.fit(df).transform(df)
        return res.select("entities").collect()[0][0]

    def get_entities(self, data: list):
        list_entities = [self._create_entity_dict(row) 
                         for row in data]
        return list_entities
    
    def _create_entity_dict(self, row):
        print(row)
        return {
            "entity": row.result,
            "ent_start": row.begin,
            "ent_end": row.end,
            "ent_type": row.metadata['entity'],
            "confidence": row.metadata['confidence']
            }

    def get_pipeline(self):
        document_assembler = DocumentAssembler() \
        .setInputCol('text') \
        .setOutputCol('document')

        tokenizer = Tokenizer() \
        .setInputCols(['document']) \
        .setOutputCol('token')

        # embeddings = DistilBertEmbeddings.load(f"../data/assets/distilbert-base-cased_spark_nlp") # notebook
        embeddings = DistilBertEmbeddings.load(f"data/assets/distilbert-base-cased_spark_nlp")\
            .setInputCols(["document",'token'])\
            .setOutputCol("embeddings")

        movie_ner_model = NerDLModel.pretrained(
        'ner_mit_movie_simple_distilbert_base_cased', 'en') \
        .setInputCols(['document', 'token', 'embeddings']) \
        .setOutputCol('ner')

        ner_converter = NerConverter() \
        .setInputCols(['document', 'token', 'ner']) \
        .setOutputCol('entities')

        pipeline = Pipeline(stages=[
            document_assembler, 
            tokenizer,
            embeddings,
            movie_ner_model,
            ner_converter
        ])

        return pipeline
    


    

if __name__ == "__main__":
    example = "I want to watch the movie Harry Potter and the Chamber of Secrets. Can you help me find it?"
    # p = MovieEntityProcessor()
    # [Row(annotatorType='chunk', begin=26, end=64, result='Harry Potter and the Chamber of Secrets', metadata={'sentence': '0', 'chunk': '0', 'entity': 'TITLE', 'confidence': '0.8125714'}, embeddings=[])]
    # [{'ent_start': 26, 'ent_end': 64, 'enntity': 'Harry Potter and the Chamber of Secrets', 'ent_type': 'TITLE', 'confidence': '0.8125714'}]    
    
    p = EntityProcessor()
    # [Row(annotatorType='chunk', begin=26, end=41, result='Harry Potter and', metadata={'sentence': '0', 'chunk': '0', 'entity': 'MISC', 'confidence': '0.858661'}, embeddings=[]), Row(annotatorType='chunk', begin=47, end=64, result='Chamber of Secrets', metadata={'sentence': '0', 'chunk': '1', 'entity': 'MISC', 'confidence': '0.9591062'}, embeddings=[])]
    # [{'enntity': 'Harry Potter and', 'ent_start': 26, 'ent_end': 41, 'ent_type': 'MISC', 'confidence': '0.858661'}, {'enntity': 'Chamber of Secrets', 'ent_start': 47, 'ent_end': 64, 'ent_type': 'MISC', 'confidence': '0.9591062'}]
    
    p.process(example)
    print(p.annotations)
    print(p.entities)

 