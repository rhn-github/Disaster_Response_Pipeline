import sys
# for loading data
import pandas as pd
import sqlite3
from sqlalchemy import create_engine

# for tokenizing
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
nltk.download(['punkt', 'wordnet', 'stopwords'])

# for pipeline build
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier

# for pipeline train
from sklearn.model_selection import train_test_split

# for model test
from sklearn.metrics import classification_report

# for gridcv
import numpy as np
from sklearn.model_selection import GridSearchCV

# for model export
import pickle

def load_data():
    # Input
    # - read csv files
    # Output
    # - X, y, category names for use in later functions
    database_filepath = sys.argv[1]
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql("SELECT * FROM DisasterResponse", engine)
    X = df.message.values
    y = df.iloc[:, 5:].values
    category_names = list(df.columns)[5:]
    return X, y, category_names


def tokenize(text):
    # tokenize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text)
    tokens = word_tokenize(text)     
    tokens = [t for t in tokens if t not in stopwords.words("english")]                 
    # initiate lemmatizer
    lemmatizer = WordNetLemmatizer() 
    # iterate through each token
    clean_tokens = []
    for tok in tokens:               
        # lemmatize, normalize case, and remove leading/trailing white space
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()     
        clean_tokens.append(clean_tok)

    return clean_tokens

def build_model():
    # Input
    # - no user input, pre-definitions
    # Output
    # - GridSearchCV ML Model
    # define basic pipeline model
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    # define optimisation parameters
    parameters = {
        'clf__estimator__n_estimators': [50, 200],
        'clf__estimator__min_samples_split': [2, 4]
        }
    # define GridSearchCV optimisation
    model = GridSearchCV(pipeline, param_grid=parameters, cv = 2)
    
    # secondary basic pipeline model 
    # trains quicker than GridSearchCV, used for debugging rest of code
    # return commented out in production version as GridSearchCV to be used
    #model = pipeline

    return model


def evaluate_model(model, X_test, y_test, category_names):
    # Input
    # - model from build_model() function.
    # - X_test from main script.
    # Output
    # - classification_report for model
    # predict on test data
    y_pred = model.predict(X_test)
    # print report    
    print(classification_report(y_test,y_pred,target_names=category_names))

def save_model(model, model_filepath):
    # Input
    # - model from build_model() function.
    # Output
    # - exported pkl file for model
    model_filepath = sys.argv[2]
    model = build_model()
    pickle.dump(model,open(model_filepath, "wb"))
    
def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data()
        
        print('Splitting data into train and test sets...')
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training Classifier...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')

if __name__ == '__main__':
    main()
