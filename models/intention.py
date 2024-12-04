import re

class IntentionDetection:
    def __init__(self):
        ''''''

    def detect_intention(self, query: str):
        if "recommend" in query.lower():
            return "recommend"
        elif self._detect_multimedia(query.lower()):
            return "multimedia"
        else:
            return "query"
        
    def _detect_multimedia(self, query: str):
        multimedia_patterns = [
        # Direct image requests
        r'(show|display|give|find|demonstrate|get|describe|see)(a)?\s*(picture|photo|image|look|depiction|representation|face|portrait|headshot|appearance|visual|like)*',
        r'(what|how)\s*(like|look|appear|seem|resemble|be|show|display|depict|represent|portray|picture|photo|image|visual|face|portrait|headshot|appearance|visual|like)*',
        ]
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in multimedia_patterns)
        
if __name__ == "__main__":
    id = IntentionDetection()
    print(id.detect_intention("Who is the director of Good Will Hunting?"))
