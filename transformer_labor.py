Fimport pandas as pd
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from textblob import TextBlob
from joblib import Parallel, delayed
import re, nltk
from sklearn.base import BaseEstimator, TransformerMixin
from scipy.sparse import csr_matrix
#import aunsight_connections as au_con

nltk.download('punkt')
ps = SnowballStemmer("english")
regex_string = '( what | is | are | was | did | where | which | when | were | will | do | have | has | does | can | any | type | pre-id | pre id)'
def tokenize_terms(text):
    text = text.lower()
    chunks = re.split(regex_string, text)
    tokens = [chunks[0]]
    for i in range(1, len(chunks), 2):
        tokens.append(chunks[i] + chunks[i+1])
    return tokens

def tokenize_mods(text):
    indices = [3,5,-3, len(text)]
    chunks = [text[0:i] for i in indices]
    return chunks

def get_preid(text):
    text = text.lower()
    chunks = re.split("pre.id", text)
    if len(chunks) == 1:
        parts = "NO PREDICTION"
    else:
        parts = chunks[1]
    return parts

# Prepares data for guided help TfidfVectorizer
class prep_mod(BaseEstimator, TransformerMixin):
    def __init__(self, column = ""):
        self.column = column
    
    def transform(self, X, *_):
        result = list(X["Engineering_Model_Number"])
        return result
    
    def fit(self, *_):
        return self


def stemmer(line):
    words = [ps.stem(w) for w in line.split()]
    return " ".join(words)

#Cleans and stems words in preparation for the word-based TfidfVectorizer
class request_to_corpus(BaseEstimator, TransformerMixin):
    def __init__(self, remove):
        self.remove = remove
    
    def transform(self, X, *_):
        text_col = list(X["Request"])
        if type(text_col) is not list:
            text_col = [text_col]
        temp = [re.sub(self.remove, '', str(i)).replace('nan', '') for i in text_col]
        result = Parallel(n_jobs=-1)(delayed(stemmer)(i) for i in temp)
        return result
    
    def fit(self, *_):
        return self

class preid_to_corpus(BaseEstimator, TransformerMixin):
    def __init__(self, remove):
        self.remove = remove
    
    def transform(self, X, *_):
        text_col = list(X["Request"].apply(lambda x: get_preid(x)))
        if type(text_col) is not list:
            text_col = [text_col]
        temp = [re.sub(self.remove, '', str(i)).replace('nan', '') for i in text_col]
        result = Parallel(n_jobs=-1)(delayed(stemmer)(i) for i in temp)
        return result
    
    def fit(self, *_):
        return self
    
# Prepares data for guided help TfidfVectorizer
class prep_work(BaseEstimator, TransformerMixin):
    def __init__(self, column = ""):
        self.column = column
    
    def transform(self, X, *_):
        result = list(X["Request"])
        return result
    
    def fit(self, *_):
        return self

# Transforms engineering model number column to a sparse dataset, which can then be joined to the other sparse sets from the TfidfVectorizers
class prod_model_to_sparse(BaseEstimator, TransformerMixin):
    def __init__(self, prefix = ""):
        self.prefix = prefix
    
    def transform(self, X, *_):
        #X['age_month'] = X['age_month'].astype('category', categories = [x for x in range(0, 120)])
        #categories = au_con.to_pandas_df("bfd18c25-f7f3-48ac-8e8b-47e4cde71bba")
        categories = pd.read_csv("../cca_training.psv", sep="|")
        categories = categories[pd.notnull(categories.Engineering_Model_Number)]
        categories = categories.Engineering_Model_Number.unique().tolist()
        categories=sorted(list(set(categories)))
        
        to_sparse_col = X["Engineering_Model_Number"].astype(pd.api.types.CategoricalDtype(categories = [x for x in categories]))
        result = csr_matrix(pd.get_dummies(to_sparse_col, prefix = self.prefix))
        return result
    
    def fit(self, *_):
        return self

class desc_to_corpus(BaseEstimator, TransformerMixin):
    def __init__(self, remove):
        self.remove = remove
    
    def transform(self, X, *_):
        text_col = list(X["Desc"])
        if type(text_col) is not list:
            text_col = [text_col]
        temp = [re.sub(self.remove, '', str(i)).replace('nan', '') for i in text_col]
        result = Parallel(n_jobs=-1)(delayed(stemmer)(i) for i in temp)
        return result
    
    def fit(self, *_):
        return self