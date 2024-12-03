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
            r'(show|display|give|find|demonstrate|get|describe|see)\s*(a)?\s*(picture|photo|image|look|depiction|representation|face|portrait|headshot|appearance|visual)\s*(of)?',
            r'visual\s*(representation|depiction)',
            r'how\s*appear',
            r'what\s*look\s*like',
        ]

        return any(re.search(pattern, query) for pattern in multimedia_patterns)
        