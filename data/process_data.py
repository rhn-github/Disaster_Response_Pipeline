import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # Function for loading & merging data from csv files
    # Inputs
        # messages.csv from messages_filepath
        # categories.csv from categories_filepath
    # Output
        # pandas df merging messages and categories
    messages =  pd.read_csv(messages_filepath)
    categories =  pd.read_csv(categories_filepath)
    df = pd.merge(messages, categories, on="id", how="outer", indicator=True)
    
    return df


def clean_data(df):
    # function for cleaning the data categories and organising them into individual columns
    # Input
    # - dataframe df
    # Output
    # - dataframe df with one column for each category
    #   and column values 1 or 0 for each row 
    #   depending on the categrories listed in each row
    #   in the original df.categories column
    # - resulting dataframe cleaned to remove duplicates
    # - column values verified to be 1 or 0 only
    
    # create a dataframe of the 36 individual category columns
    categories = df.categories.str.split(";", expand=True)
    
    # select the first row of the categories dataframe
    row = categories.iloc[0]
                      
    # extract a list of new column names for categories
    # from this row by applying a lambda function that takes everything 
    # up to the second to last character of each string with slicing
    category_colnames = row.str.split('-').str.get(0)
                      
    # rename the columns of `categories` with these names
    categories.columns = category_colnames
                      
    #Convert category values to just numbers 0 or 1
        # set each value to be the last character of the string
        # convert column from string to numeric
    for column in categories:
        categories[column] = categories[column].str.split('-').str.get(1)    
        categories[column] = pd.to_numeric(categories[column])
        
    # drop the original categories column from `df`
    df.drop(['categories'], axis=1, inplace=True)
    
    # concatenate the original dataframe with the new `categories` dataframe
    df = pd.concat([df, categories], axis=1, join="inner")
    
    # drop duplicates
    df.drop_duplicates(inplace=True)
    
    # replace values != 1 or 0:
    for column in category_colnames:
        df[column] = np.where(df[column] > 1, 1,df[column])
        df[column] = np.where(df[column] < 0, 0,df[column])
        
    return df


def save_data(df, database_filename):
    # function to save dataframe as database file
    # input dataframe df
    # output database file containing table 'DisasterResponse'
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('DisasterResponse', engine, if_exists='replace', index=False)



def main():
    # main operating function in script
    if len(sys.argv) == 4:
        # define filepaths using contents execution statement
        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
