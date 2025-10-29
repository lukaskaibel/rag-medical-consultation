import re

def normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = text.lower()
    text = re.sub(r"[^a-zäöüß]", "", text)
    
    return text