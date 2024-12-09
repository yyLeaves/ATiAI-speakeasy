import pandas as pd
import numpy as np

ANSWER_CROWD_TEMPLATE = [
    f"[crowd] Our crowd suggests, the answer is <answer>",
    f"[crowd] Our voters think the answer should be <answer>",
    f"[crowd] The people have spoken it should be <answer>",
    f"[crowd] Just checked in the crowdsourcing kitchen, the answer should be <answer>",
    f"[crowd] Our crowd backup tells me the answer is <answer>",
]

class CrowdsourcingProcessor:
    def __init__(self, crowd_data_path="data/crowd_data.tsv", entity_labels_path="data/df_ent.pkl", relation_labels_path="data/df_rel_extend_embed.pkl"):
        self.crowd_data_path = crowd_data_path
        self.entity_labels_path = entity_labels_path
        self.relation_labels_path = relation_labels_path
        self.crowdsourcing_data = self.filter_malicious_workers(self.load_crowdsourcing_data(crowd_data_path))
        self.entity_labels = self.load_entity_labels(entity_labels_path)
        self.relations_labels = self.load_entity_labels(entity_labels_path)

    def load_crowdsourcing_data(self, file_path):
        return pd.read_csv(file_path, sep='\t')

    def load_entity_labels(self, file_path):
        return pd.read_pickle(file_path)

    # def translate_entity_to_wikidata_id(self, entity_name):
    #     """
    #     Translate an entity name or list of entity names to Wikidata IDs.
    #     This will return the first successful translation.
    #     """
    #     if isinstance(entity_name, list):  # If it's a list, process each entity individually
    #         for entity in entity_name:
    #             entity_id = self.translate_entity_to_wikidata_id(entity)
    #             if entity_id:  # Return the first valid translation
    #                 return entity_id
    #         return None  # If no valid translation is found
    #     else:
    #         # If it's a single entity, proceed as before
    #         entity_name = entity_name.lower().strip()
    #         result = self.entity_labels[self.entity_labels['label'].str.lower() == entity_name]
            
    #         if not result.empty:
    #             return 'wd:' + result.iloc[0]['uri'].split('/')[-1]  # Assuming URI is the last part of the URL
    #         return None
    def translate_entity_to_wikidata_id(self, entity_name):
        """
        Translate an entity name or list of entity names to Wikidata IDs.
        This will return the first successful translation.
        """
        # If the entity is a dictionary, extract the 'entity' value
        if isinstance(entity_name, dict):
            entity_name = entity_name.get('entity', '').lower().strip()

        # If the entity_name is a list, process each entity individually
        if isinstance(entity_name, list):
            for entity in entity_name:
                # Call translate_entity_to_wikidata_id recursively to process the list elements
                entity_id = self.translate_entity_to_wikidata_id(entity)
                if entity_id:  # Return the first valid translation
                    return entity_id
            return None  # If no valid translation is found
        else:
            # If it's a single string entity, proceed as before
            entity_name = entity_name.lower().strip()
            result = self.entity_labels[self.entity_labels['label'].str.lower() == entity_name]
            
            if not result.empty:
                # Return the Wikidata ID
                return 'wd:' + result.iloc[0]['uri'].split('/')[-1]  # Assuming URI is the last part of the URL
            return None

    def filter_malicious_workers(self, df, min_approval_rate=60, min_work_time=20):
        df['LifetimeApprovalRate'] = pd.to_numeric(df['LifetimeApprovalRate'].str.rstrip('%'), errors='coerce')
        df['WorkTimeInSeconds'] = pd.to_numeric(df['WorkTimeInSeconds'], errors='coerce')
        df = df.dropna(subset=['LifetimeApprovalRate', 'WorkTimeInSeconds'])
        return df[(df['LifetimeApprovalRate'] >= min_approval_rate) & (df['WorkTimeInSeconds'] >= min_work_time)]

    def retrieve_crowdsourced_object(self, subject, predicate, crowdsourcing_data):
        filtered = crowdsourcing_data[
            (crowdsourcing_data['Input1ID'] == subject) &
            (crowdsourcing_data['Input2ID'] == predicate)
        ]
        if filtered.empty:
            return None

        # Gather vote counts for both 'CORRECT' and 'INCORRECT' answers.
        support_votes = filtered[filtered['AnswerLabel'] == 'CORRECT']['Input3ID'].count()
        reject_votes = filtered[filtered['AnswerLabel'] == 'INCORRECT']['Input3ID'].count()
        total_votes = support_votes + reject_votes
        agreement_score = support_votes / total_votes if total_votes > 0 else 0

        # Use the mode of 'Input3ID' from all entries if there are any votes.
        if total_votes > 0:
            majority_answer = filtered['Input3ID'].mode()[0]
        else:
            majority_answer = "UNKNOWN"  # If no votes, return "UNKNOWN" to indicate lack of data.

        return majority_answer, agreement_score, {'support': support_votes, 'reject': reject_votes}

    def get_hit_type_id(self, df, subject, predicate):
        hit_type_id_row = df[(df['Input1ID'] == subject) & (df['Input2ID'] == predicate)]
        if not hit_type_id_row.empty:
            return hit_type_id_row['HITTypeId'].iloc[0]
        return None

    def calculate_inter_rater_agreement(self, df, hit_type_id):
        filtered = df[df['HITTypeId'] == hit_type_id]
        if filtered.empty:
            return None
        grouped = filtered.groupby(['HITId', 'AnswerLabel']).size().unstack(fill_value=0)
        grouped = grouped.reindex(columns=['CORRECT', 'INCORRECT'], fill_value=0)
        ratings_matrix = grouped.values
        return self.fleiss_kappa(ratings_matrix)

    def fleiss_kappa(self, ratings):
        n_items, n_categories = ratings.shape
        n_raters = ratings.sum(axis=1)[0]
        p = ratings.sum(axis=0) / (n_items * n_raters)
        P = ((ratings ** 2).sum(axis=1) - n_raters) / (n_raters * (n_raters - 1))
        P_bar = P.mean()
        P_e = (p ** 2).sum()
        kappa = (P_bar - P_e) / (1 - P_e)
        return kappa

    def generate_answer(self, graph_result, subject, predicate):
        """
        Generate a human-readable answer using crowdsourcing data.
        """
        subject = self.translate_entity_to_wikidata_id(subject)
        predicate = "wdt:"+predicate
        obj_data = self.retrieve_crowdsourced_object(subject, predicate, self.crowdsourcing_data)
        if obj_data:
            obj, agreement, counts = obj_data
            support_votes = counts['support']
            reject_votes = counts['reject']

            fleiss_score = self.calculate_inter_rater_agreement(self.crowdsourcing_data, self.get_hit_type_id(self.crowdsourcing_data, subject, predicate))
            fleiss_score = fleiss_score if fleiss_score is not None else 0.000
            distribution = f"{support_votes} support votes, {reject_votes} reject votes"

            # Use the correct answer ID directly from obj, assuming obj is the Input3ID or its translation
            object_readble = self.translate_res(obj)
            # print("THIS IS CROWD!", support_votes)
            # Print the retrieved Input3ID or the translated label
            # Randomly select a lively template from the list
            crowd_response_template = np.random.choice(ANSWER_CROWD_TEMPLATE).replace("<answer>", object_readble)

            return (f"{crowd_response_template}\n"
                    f"[Crowd, inter-rater agreement {fleiss_score:.3f}, "
                    f"The answer distribution for this specific task was {distribution}].")

        return None

    def translate_res(self, res):
        """
        Translate a Wikidata entity ID to a human-readable label if necessary.
        """
        print(f"Translating: {res}")
        if res.startswith("wd:Q"):
            ent_id = res.split(':')[-1]
            label_row = self.entity_labels[self.entity_labels['uri'].str.contains(ent_id)]
            if not label_row.empty:
                lbl = label_row.iloc[0]['label']
                print(f"Translated: {lbl}")
                return lbl
            return "UNKNOWN"
        return res  # Return as-is if it does not start with 'wd:Q'


# Example usage
if __name__ == "__main__":

    processor = CrowdsourcingProcessor()

    subject = ['X-Men: First Class']  # X-Men: First Class
    predicate = "P1431"  # executive producer


    # entity = processor.translate_entity_to_wikidata_id("X-Men: First Class")
    # print(entity)
    # answer = processor.generate_answer("hello", subject, predicate)
    # print(answer)
    # merge = ResponseMerger()
    # merged = merge.merge_responses(answer, answer)
    # print(merged)
