# src/translate.py

class TranslatorEnToTe:
    def __init__(self):
        print("⚡ Using ultra-fast rule-based Telugu translator (no models)")
        
        # common programming explanation phrases
        self.map = {
            "the code": "కోడ్",
            "function": "ఫంక్షన్",
            "loops": "లూప్ చేస్తుంది",
            "loop": "లూప్",
            "condition": "పరిస్థితి",
            "checks": "తనిఖీ చేస్తుంది",
            "returns": "తిరిగి ఇస్తుంది",
            "result": "ఫలితం",
            "value": "విలువ",
            "input": "ఇన్‌పుట్",
            "processes": "ప్రాసెస్ చేస్తుంది",
            "item": "అంశం",
            "items": "అంశాలు",
            "list": "జాబితా",
            "number": "సంఖ్య",
            "increments": "పెంచుతుంది",
            "decrements": "తగ్గిస్తుంది",
            "runs": "నడుస్తుంది"
        }

    def translate(self, text: str) -> str:
        t = text.lower()
        for eng, tel in self.map.items():
            t = t.replace(eng, tel)

        # Capitalize first letter
        if t:
            t = t[0].upper() + t[1:]
        return t
