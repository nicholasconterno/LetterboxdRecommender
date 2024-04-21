import sqlite3
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from tqdm import tqdm


# Connect to SQLite database and load movie details
def connect_to_db_and_read_and_filter():
    """
    Connect to SQLite database and load movie details. Filter out movies and users with fewer than a certain number of ratings.
    """
    # Connect to your SQLite database
    conn = sqlite3.connect('my_letterboxd_data.db')

    # Load ratings data
    query = """
    SELECT username, movie_name, rating
    FROM users
    """
    # Load the data into a DataFrame
    ratings_df = pd.read_sql(query, conn)
    ratings_df.dropna(subset=['rating'], inplace=True)
    ratings_df['rating'] = ratings_df['rating'].astype(float)
    ratings_df['username'] = ratings_df['username'].astype(str)
    ratings_df['movie_name'] = ratings_df['movie_name'].astype(str)

    query_movie_details = """
    SELECT letterboxd_slug, movie_name, director, actors, genres
    FROM film_details_small
    """
    movie_details_df = pd.read_sql(query_movie_details, conn)

#rename movie_name to  real_movie_name
    movie_details_df.rename(columns={'movie_name': 'real_movie_name'}, inplace=True)
    # rename letterboxd_slug to movie_name
    movie_details_df.rename(columns={'letterboxd_slug': 'movie_name'}, inplace=True)
    
    # Example of filtering out movies and users with fewer than a certain number of ratings
    min_movie_ratings = 25 # Movies with fewer than 10 ratings
    min_user_ratings = 50 # Users with fewer than 5 ratings
    print(len(ratings_df))
    filtered_ratings = ratings_df.groupby('movie_name').filter(lambda x: len(x) >= min_movie_ratings)
    filtered_ratings = filtered_ratings.groupby('username').filter(lambda x: len(x) >= min_user_ratings)
    # print('hello')
    # Proceed with the filtered_ratings DataFrame
    ratings_df = filtered_ratings
    print(len(ratings_df))
    conn.close()
    return ratings_df, movie_details_df

"""
Split the data into training and testing sets, create a user-movie ratings matrix, and apply SVD to reduce the dimensionality of the data."""
def split_data(ratings_df):
    # train test split usernames
    train_users, test_users = train_test_split(ratings_df['username'].unique(), test_size=0.2, random_state=42)

    # split the data into training and testing
    test_data = ratings_df[ratings_df['username'].isin(test_users)]
    ratings_df = ratings_df[ratings_df['username'].isin(train_users)]
    return ratings_df, test_data

"""
Create a user-movie ratings matrix, apply SVD to reduce the dimensionality of the data, and compute similarity scores between users."""
def create_user_movie_matrix(ratings_df):
    # Create a user-movie ratings matrix
    user_movie_ratings = ratings_df.pivot_table(index='username', columns='movie_name', values='rating').fillna(0)

    # Convert to sparse matrix
    ratings_matrix = csr_matrix(user_movie_ratings.values)

    # Apply SVD
    svd = TruncatedSVD(n_components=20) # You can adjust the number of components
    matrix_reduced = svd.fit_transform(ratings_matrix)

    # Compute similarity scores
    user_similarity = cosine_similarity(matrix_reduced)
    return user_movie_ratings, user_similarity, svd, matrix_reduced


"""
Predict the top movies for a given user based on the user-movie ratings matrix and user similarity scores."""
def predict_top_movies(user_index, user_movie_ratings, user_similarity, top_k=10):
    # Compute similarity scores with other users
    similarity_scores = list(enumerate(user_similarity[user_index]))
    # Sort users by similarity score in descending order (most similar first)
    similarity_scores = sorted(similarity_scores, key=lambda x: x[1], reverse=True)

    # Get indices of top_k similar users (excluding the user itself which is at index 0)
    top_users_indices = []
    for i in (range(1, 1000)):  # Considering top 999 similar users after excluding the user itself
        top_users_indices.append(similarity_scores[i][0])
    
    # Select the ratings of these top users
    top_users_ratings = user_movie_ratings.iloc[top_users_indices]

    # Filter movies where less than 5 users rated it (non-zero ratings)
    valid_movies = top_users_ratings.apply(lambda x: x > 0).sum(axis=0) >= 5
    top_users_ratings = top_users_ratings.loc[:, valid_movies]

    # Calculate the mean of ratings, ignoring zeros
    recommended_movies = top_users_ratings.apply(lambda x: np.mean(x[x > 0]), axis=0)

    # remove movies not in the movie_details_df
    recommended_movies = recommended_movies[recommended_movies.index.isin(movie_details_df['movie_name'])]
    print('ayoo')
    # remove movies that are documentaries
    recommended_movies = recommended_movies[~recommended_movies.index.isin(movie_details_df[movie_details_df['genres'].str.contains('Documentary')]['movie_name'])]

    print('ayoo2')

    # remove movies the user has already rated
    user_rated_movies = user_movie_ratings.iloc[user_index]
    recommended_movies = recommended_movies[~recommended_movies.index.isin(user_rated_movies[user_rated_movies > 0].index)]

    # Sort the average ratings in descending order and select the top_k movies
    recommended_movies = recommended_movies.sort_values(ascending=False)
    return recommended_movies[:top_k]


"""
Predict the top movies for a new user based on the user-movie ratings matrix and SVD model."""
def predict_movies_for_new_user(new_user_ratings,user_movie_ratings,matrix_reduced,movie_details_df,user_index,svd, top_k=10):
    # Integrate new user ratings into the existing user-movie matrix
    # Create a Series from the new user ratings, reindexing to match the columns of the existing matrix
    new_user_series = pd.Series(new_user_ratings).reindex(user_movie_ratings.columns).fillna(0)
    
    # Append this user to the existing matrix and transform using the existing SVD model
    new_user_vector = svd.transform(csr_matrix(new_user_series.values.reshape(1, -1)))

    # Compute cosine similarity between this new user and all other users
    new_user_similarity = cosine_similarity(new_user_vector, matrix_reduced).flatten()

    # Exclude the new user's self-comparison and get indices of top similar users
    top_users_indices = np.argsort(-new_user_similarity)[1:1000]
    top_users_ratings = user_movie_ratings.iloc[top_users_indices]

    # Filter movies where less than 5 users rated it (non-zero ratings)
    valid_movies = top_users_ratings.apply(lambda x: x > 0).sum(axis=0) >= 5
    top_users_ratings = top_users_ratings.loc[:, valid_movies]

    # Calculate the mean of ratings, ignoring zeros
    recommended_movies = top_users_ratings.apply(lambda x: np.mean(x[x > 0]), axis=0)

    # remove movies not in the movie_details_df
    recommended_movies = recommended_movies[recommended_movies.index.isin(movie_details_df['movie_name'])]
    print('ayoo')
    # remove movies that are documentaries
    recommended_movies = recommended_movies[~recommended_movies.index.isin(movie_details_df[movie_details_df['genres'].str.contains('Documentary')]['movie_name'])]

    print('ayoo2')

    # remove movies the user has already rated
    user_rated_movies = user_movie_ratings.iloc[user_index]
    # recommended_movies = recommended_movies[~recommended_movies.index.isin(user_rated_movies[user_rated_movies > 0].index)]

    # Sort the average ratings in descending order and select the top_k movies
    recommended_movies = recommended_movies.sort_values(ascending=False)
    return recommended_movies[:top_k]

"""
Evaluate the SVD model by predicting top movies for test users and calculating the mean absolute error, mean squared error, and root mean squared error."""
def test_svd_model(test_data, user_movie_ratings, svd, user_similarity, movie_details_df):
    mae = []
    mse = []
    for user in tqdm(test_data['username'].unique()):
        user_index = user_movie_ratings.index.get_loc(user)
        test_user_ratings = test_data[test_data['username'] == user]
        top_movies = predict_top_movies(user_index, user_movie_ratings, user_similarity)
        # get error for the user
        for movie in test_user_ratings['movie_name']:
            if movie in top_movies.index:
                mae.append(mean_absolute_error([test_user_ratings[test_user_ratings['movie_name'] == movie]['rating']], [top_movies[movie]]))
                mse.append(mean_squared_error([test_user_ratings[test_user_ratings['movie_name'] == movie]['rating']], [top_movies[movie]]))
            else:
                mae.append(mean_absolute_error([test_user_ratings[test_user_ratings['movie_name'] == movie]['rating']], [0]))
                mse.append(mean_squared_error([test_user_ratings[test_user_ratings['movie_name'] == movie]['rating']], [0]))

    print(f"Mean Absolute Error: {np.mean(mae)}")
    print(f"Mean Squared Error: {np.mean(mse)}")
    # print rmse
    print(f"Root Mean Squared Error: {np.sqrt(np.mean(mse))}")

"""
Main function to connect to the database, read and filter the data, split the data, create the user-movie ratings matrix, and test the SVD model."""
if __name__ == "__main__":
    ratings_df, movie_details_df = connect_to_db_and_read_and_filter()
    ratings_df, test_data = split_data(ratings_df)
    user_movie_ratings, user_similarity, svd, matrix_reduced = create_user_movie_matrix(ratings_df)
    test_svd_model(test_data, user_movie_ratings, svd, user_similarity, movie_details_df)
    
