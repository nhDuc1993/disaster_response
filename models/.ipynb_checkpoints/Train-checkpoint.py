import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

import pandas as pd
import numpy as np

import re
from sqlalchemy import create_engine

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.base import BaseEstimator,TransformerMixin

import pickle


def load_data():
    """
    Load data from database
    """
    
    # Load data from database into a dataframe
    filepath = "../data/DisasterResponse.db"
    engine = create_engine('sqlite:///' + filepath)
    
    df = pd.read_sql_table('DisasterResponse', engine)
    
    # Clean data: drop 'child_alone' column and fix 'related' colummn
    df.drop('child_alone', axis=1, inplace=True)
    df['related'] = df['related'].apply(lambda x: 1 if x==2 else x)
    
    X = df['message']
    y = df.loc[:,'related':]
    
    return X,y


def tokenize(text):
    """
    Tokenize the text
    
    - Input: text message
    - Output: tokens from the text message
    """
    
    # Create a regex of url strings:
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    
    # Find all occurences of url strings:
    detected_urls = re.findall(url_regex, text)
    
    # Replace all url strings with 'urlplaceholder'
    for url in detected_urls:
        text = text.replace(url, 'urlplaceholder')
        
    # Extract the token
    tokens = word_tokenize(text)
    
    # Lemmatize
    lemmatize = WordNetLemmatizer()
    
    # Return clean tokens:
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatize.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Build StartingVerbExtractor transformer to extract the starting verb
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


def build_model():
    """
    Build a pipeline
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', RandomForestClassifier())
    ])
    return pipeline


def display_results(y_test, y_pred):
    """
    Display evaluation scores
    """
    labels = np.unique(y_pred)    
    accuracy = (y_pred == y_test).mean()

    print("Labels:", labels)
    print("Accuracy:", accuracy)

def export_as_pickle(pipeline):
    """
    Export model as pickle
    """
    pickle.dump(pipeline,open('model.pkl', 'wb'))

def main():
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y)

    model = build_model()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    display_results(y_test, y_pred)
    
    export_as_pickle(model)

if __name__ == '__main__':
    main()

