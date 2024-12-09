import re

class IntentionDetection:
    def __init__(self):
        ''''''

    def detect_intention(self, query: str):
        query = ' '.join(query.split()) # remove extra space
        if "recommend" in query.lower():
            return "recommend"
        elif self._detect_multimedia(query.lower()):
            return "multimedia"
        else:
            return "query"
        
    def _detect_multimedia(self, query: str):
        multimedia_patterns = [
        r'.{0,}(show|display|give|find|demonstrate|get|describe|see).{0,}(picture|photo|image|look|depiction|representation|face|portrait|headshot|appearance|visual|like).{0,}',
        r'.{0,}(what|how).{0,}(looks like|looked like|look like|look|appear|seem|resemble|be|show|display|depict|represent|portray|picture|photo|image|visual|face|portrait|headshot|appearance|visual).{0,}',
        ]
        return any(re.search(pattern, query, re.IGNORECASE) for pattern in multimedia_patterns)
        
# if __name__ == "__main__":
#     id = IntentionDetection()
#     print(id.detect_intention("Show me the director of 12 monkeys image"))

if __name__ == "__main__":
    id = IntentionDetection()
    print(id.detect_intention("Who is the director of Good Will Hunting?"))