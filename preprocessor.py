import re
import pandas as pd 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

stop_words = set(stopwords.words("english"))
NON_ALPHANUM = re.compile(r'[^\w\s]')

def clean_review(row):
    cleaned_text = NON_ALPHANUM.sub('', row.lower())
    filtered_tokens = [word for word in cleaned_text.split() if word not in stop_words]
    return " ".join(filtered_tokens)

def load_data(filepath):
    df = pd.read_csv(filepath)
    df["cleaned_review"] = df["review"].apply(clean_review)
    df = df.drop(columns = ["review"])
    return df

y = load_data("IMDB_DATASET.csv")
print(y.head())