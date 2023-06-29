import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
import re
# # no cors

nltk.download('punkt')#C:\Users\pranj\OneDrive\djangio\Compfox-Assesment\AI-Features\Search-API\Dockerfile
nltk.download('stopwords')
nltk.download('wordnet')
# from fastapi.middleware.cors import CORSMiddleware

# origins = [
#     "http://localhost",
#     "http://localhost:3000",
#     "http://localhost:8000",
#     "http://localhost:8080",
#     "https://staging.compfox.io"
# ]



# app = FastAPI()
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=origins,
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


def lemmatize_and_clean_text(text):
    tokens = word_tokenize(text)

    lemmatizer = WordNetLemmatizer()

    cleaned_tokens = []
    for token in tokens:
        token = token.lower()
        token = lemmatizer.lemmatize(token)
        if token not in stopwords.words("english") and len(token) > 2:
            cleaned_tokens.append(token)
    
    return cleaned_tokens

def highlight_words(word_list, text, output_file):
    pattern = '|'.join(r'\b{}\b'.format(re.escape(word)) for word in word_list)
    highlighted_text = re.sub(pattern, r'<span style="background-color: yellow">\g<0></span>', text, flags=re.IGNORECASE)

    with open(output_file, 'w') as file:
        file.write('<html><body>{}</body></html>'.format(highlighted_text))

    print("Highlighted text saved in '{}'.".format(output_file))

    return highlighted_text 

def get_highlighted_phrases(word_list, text):
    pattern = '|'.join(r'\b{}\b'.format(re.escape(word)) for word in word_list)
    highlighted_phrases = []

    def highlight(match):
        start = max(0, match.start() - 160)
        end = min(len(text), match.end() + 160)
        return text[start:end]

    matches = re.finditer(pattern, text, flags=re.IGNORECASE)
    for match in matches:
        highlighted_phrases.append(highlight(match))

    return highlighted_phrases