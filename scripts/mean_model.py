import sqlite3
import pandas as pd
import numpy as np
import tensorflow as tf
import sys
print(tf.__version__)
print(sys.executable)
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers

import tensorflow_recommenders as tfrs
from sklearn.model_selection import train_test_split

def connect_to_db_and_read():
    """
    Connect to SQLite database and load movie details."""
    # Connect to SQLite database and load movie details
    conn = sqlite3.connect('../data/my_letterboxd_data.db')

    # Load ratings data
    query_ratings = """
    SELECT username, movie_name, rating
    FROM users
    """
    ratings_df = pd.read_sql(query_ratings, conn)
    # Check if the DataFrame is empty or if specific columns are empty
    print(ratings_df.head())
    print("Data types:", ratings_df.dtypes)
    print("Count of non-NA values:\n", ratings_df.count())
    # Load movie details
    query_movie_details = """
    SELECT letterboxd_slug, movie_name, director, actors, genres
    FROM film_details_small
    """
    movies_details_df = pd.read_sql(query_movie_details, conn) # REAL MOVIE NAME
    # rename columns from movie_details_df
    movies_details_df.rename(columns={'movie_name': 'real_movie_name'}, inplace=True)
    movies_details_df.rename(columns={'letterboxd_slug': 'movie_name'}, inplace=True)
    conn.close()
    return ratings_df, movies_details_df

def get_movie_names(ratings_df):
    # get list of unique movie names sorted by count of ratings from ratings_df
    movie_names = ratings_df['movie_name'].value_counts().index.tolist()
    return movie_names

def preprocess(ratings_df, movies_details_df):
    """
    Preprocess the ratings and movie details DataFrames by filling missing values and merging them."""
    # Data preprocessing
    ratings_df['rating'] = ratings_df['rating'].astype(float)
    ratings_df = ratings_df.fillna(-1)
    movies_details_df.fillna('', inplace=True)  # Handle missing values
    # print(ratings_df.head(2))
    # Merge ratings with movie details
    df = pd.merge(ratings_df, movies_details_df, on='movie_name', how='left')
    return df

def create_map_movie_to_average_rating(df):
    # Group by 'movie_name' and calculate the mean of 'rating' for each group
    movie_to_rating = df.groupby('movie_name')['rating'].mean().to_dict()
    return movie_to_rating

# create a function that returns the top n movies for a user that they have not rated yet using get_movie_recommendations 
def get_user_recommendations(username, df, n_recommendations=10):
    # get all the movies that the user has not rated
    user_rated_movies = df[df['username'] == username]['movie_name']
    all_movies = df['movie_name'].unique()
    # get the list of movies that the user has not rated
    movies_to_rate = np.setdiff1d(all_movies, user_rated_movies)
    recommendations = []
    # get the average rating for each movie
    m_to_r = create_map_movie_to_average_rating(df)
    for movie in (movies_to_rate):
        # get the average rating for the movie
        recommendations.append((movie, m_to_r[movie]))
    # sort the recommendations by the rating in descending order
    recommendations = sorted(recommendations, key=lambda x: x[1], reverse=True)
    return recommendations[0:n_recommendations]

# create a function that returns the top n movies for a user that they have not rated yet using get_movie_recommendations
def get_mean_recommendations(username):
    """
    Get the top 10 movie recommendations for a user based on the mean rating of the movies."""
    ratings_df, movies_details_df = connect_to_db_and_read()
    df = preprocess(ratings_df, movies_details_df)
    recommendations = get_user_recommendations(username, df)
    return recommendations

if __name__ == "__main__":
    recommendations = get_mean_recommendations('nconterno')
    print(recommendations)
