import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """
    Load data
    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)

    df = messages_df.merge(categories_df, left_on='id', right_on='id')

    return df


def clean_data(df):
    """
    Clean data
    """
    categories = df['categories'].str.split(';', expand=True)

    row = categories.iloc[0,:]
    category_colnames = row.apply(lambda x: x.partition('-')[0])
    categories.columns = category_colnames
    
    for column in categories.columns:
        categories[column] = categories[column].apply(lambda x: x.partition('-')[2])
        categories[column] = categories[column].astype('int32')
    
    df.drop('categories', axis=1, inplace=True)
    df = pd.concat([df, categories], axis=1)
    
    df['related'] = df['related'].apply(lambda x: 1 if x==2 else x)
    
    df.drop_duplicates(inplace=True)
    return df
    
def save_data(df):
    """
    export data
    """
    engine = create_engine('sqlite:///DisasterResponse.db')
    df.to_sql('DisasterResponse', engine, index=False, if_exists='replace')

    
def main():
    df = load_data(r'disaster_messages.csv', r'disaster_categories.csv')
    
    df = clean_data(df)
    
    save_data(df)

    
if __name__ == '__main__':
    main()

