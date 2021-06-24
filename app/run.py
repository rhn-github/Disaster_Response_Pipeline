import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar, Pie
from sklearn.externals import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('DisasterResponse', engine)

# load model
# real model from GridSearchCV classifier
model = joblib.load("../models/classifier.pkl")
# test model from pipeline classifier
#model = joblib.load("../models/classifier_p.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # dataframe summing values of category columns
    category = df.iloc[:-1, 4:]
    category_sum = category.sum().sort_values(ascending = False)
    categories = list(category_sum.index)
    
    # top 10
    category_sum_10 = category_sum[:10]
    categories_10 = categories[:10]
    # rest
    category_sum_rest = category_sum[10:].sum()
    # create new series top 10 classes plu the rest
    values_top_rest = pd.Series(category_sum_10).append(pd.Series(category_sum_rest))
    categs_top_rest = pd.Series(categories_10).append(pd.Series(['rest']))
    
    
    # create visuals
    graphs = [
        # Pie Chart Genres
        {
            'data': [
                Pie(
                    labels=genre_names,
                    values=genre_counts,
                    hoverinfo='label+percent'
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                
            }
        },
        # Bar Chart Categories Ranking
        {
            'data': [
                Bar(
                    x=categories,
                    y=category_sum
                )
            ],

            'layout': {
                'title': 'Ranking of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': ""
                }
            }
        },
        # Pie Chart top 10 Categories vs Rest
        {
            'data': [
                Pie(
                    labels=categs_top_rest,
                    values=values_top_rest,
                    hoverinfo='label+percent'
                )
            ],

            'layout': {
                'title': 'Top 10 Categories vs Rest',
                
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
