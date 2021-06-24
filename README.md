# Disaster Response Pipeline#
Udacity Nanodegree Data Scientist, Disaster Response Pipeline Project

[1. Introduction](#intro)<br>
[2. Repository Contents](#contents)<br>
[3. Instructions for use](#instructs)<br>
[4. Notes on GridSearchCV Parameter Selection](#params)<br>
[5. Credits](#credits)<br>

<a id='intro'></a>
### Introduction:
In this project, I have analyzed disaster data from [Figure Eight](https://www.figure-eight.com/) to build a model for an API that classifies disaster messages.<br>
(Note that Figure Eight is now owned by [Appen](https://appen.com/)).

The output of the analysis can be seen in a web app with visualisations as follows:


<a id='contents'></a>
### Repository Contents:
#### Disaster_Response_Pipeline
* README.md
  - This File
#### ...\data
* disaster_messages.csv
  - csv file containing messages communicated during real world disasters
* disaster_categories.csv
  - csv file containing the disaster catgeories relating to these messages
* process_data.py
  - a python script containing an `ETL Pipeline` that
    - reads the two csv files into a pandas dataframe
    - cleans and organizes the data
    - exports the dataframe to a database file *DisasterResponse.db*
* DisasterResponse.db
  - Sample database file created using *disaster_messages.csv, disaster_categories.csv* and *process_data.py*
#### ...\model
* train_classifier.py
  - a python script containing a `ML Pipeline` that
    - reads data from the SQLite database file *DisasterResponse.db*
    - Splits the dataset into training and test sets
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports the final model as a pickle file *classifier.pkl*
  - classifier.pkl
    - Pickle file based on basic GridSearchCV model
  - classifier_p.pkl
    - Pickle file based on basic pipeline model
#### ...\app
* run.py
  - a Flask file that runs app
#### ...\app\template
* master.html
  - main page of web app
* go.html
  - classification result page of web app
#### ...\additional_information
* ML Pipeline Preparation.html
  - Jupyter Notebook HTML Download showing result of 5 Parameter GridSearchCV optimizataion

<a id='instructs'></a>
### Instructions for Use:
1. In your workspace create a Project Directory containing 
* the sub-folders `\app`,`\data`,`\model`
* in `\app` create another subfolder `\templates`
* from this repository copy the contents of the folders `app`, `data`, and `models` to the corresponding sub-folders in the Project Directory

2. Open a workspace terminal and run the following commands in the project's root directory to set up the database *DisasterResponse.db* and model *classifier.py*. Note that sample versions of these two files are also provided in this GitHub repository.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

3. Run the following command in the app's directory to run your web app.
    `python run.py`

Accessing the WebApp if using Udacity Workspace IDE:

1. Open a new workspace terminal and run
    `env|grep WORK`

4. note the Ouput values for *SPACEID* and *SPACEDOMAIN*

3. Use these values to complete the address *`https://SPACEID-3001.SPACEDOMAIN`* and open a new Web Page in the browser with this address to access the Application.

Accessing the WebApp if using Local Machine:

1. Go to http://0.0.0.0:3001/



<a id='params'></a>
### Notes on Parameter Selection in GridSearchCV

In the python script *train_classifier.py* a GridSearchCV model was built, optimising 2 parameters:

    'clf__estimator__n_estimators': [50, 200],
    'clf__estimator__min_samples_split': [2, 4]
    
Results for 2 Parameters (may vary if models re-run)
         
| model |   | precision | recall | f1-score | support |
|----|----|----|----|----|----|
| basic pipeline | avg / total | 0.73 | 0.49 | 0.54 | 16212 |
| GridSearchCV | avg / total | 0.76 | 0.549 | 0.57 | 16441 |


During development a second parameter optimisation was attempted , optimising 5 parameters: 

    'vect__max_df': [10, 25],
    'vect__min_df': [1, 5],
    'tfidf__use_idf': [True, False],
    'clf__estimator__n_estimators': [10, 25],
    'clf__estimator__min_samples_split': [2, 4]

Results for 5 Parameters (may vary if models re-run)
| model |   | precision | recall | f1-score | support |
|----|----|----|----|----|----|
| basic pipeline | micro avg | 0.82 | 0.53 | 0.55 | 20822 |
| basic pipeline | macro avg | 0.56 | 0.20 | 0.25 | 20822 |
| basic pipeline | weighted avg | 0.76 | 0.53 | 0.57 | 20822 |
| basic pipeline | samples avg | 0.67 | 0.49 | 0.52 | 20822 |
| GridSearchCV | micro avg | 0.80 | 0.56 | 0.66 | 20822 |
| GridSearchCV | macro avg | 0.55 | 0.24 | 0.30 | 20822 |
| GridSearchCV | weighted avg | 0.74 | 0.56 | 0.60 | 20822 |
| GridSearchCV | samples avg | 0.65 | 0.50 | 0.52 | 20822 |

For the basic pipeline training time of about 5 minutes was observed, whilst for the GridSearchCV tuning 2 parameters about an hour was needed and for the GridSearchCV tuning 5 parameters 5 hours 45 minutes.

Comparing the results shows that only small increases in accuracy are achieved using GridSearchCV compared to the basic pipeline with this data, in spite of the significantly longer training times.

This would suggest that with quicker training times, the basic pipeline model would be adequate for use with this data. This is perhaps not surprising when one considers that the message data originates from victims and aid workers reporting from the chaos that is a disaster zone, and is therefore likely to be full of anomalies and inaccuracies


<a id='credits'></a>
### Credits
Credit is due to 
* [Figure Eight/Appen](https://appen.com/) for providing the Data
* Udacity for providing the learning environment and training material
* Rajat S of Udacity's Mentor teams whose guidance has helped me join the dots to understand this process...

