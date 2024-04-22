import sqlite3
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense


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

    # only keep the movies that are in the movie_details_df
    ratings_df = ratings_df[ratings_df['movie_name'].isin(movie_details_df['movie_name'])]

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

def split_data(ratings_df):
        # train test split usernames
    train_users, test_users = train_test_split(ratings_df['username'].unique(), test_size=0.2, random_state=42)

    # split the data into training and testing
    test_data = ratings_df[ratings_df['username'].isin(test_users)]
    ratings_df = ratings_df[ratings_df['username'].isin(train_users)]

    print('test data made')
    return ratings_df, test_data


def create_user_movie_ratings_matrix(ratings_df):
    # Create a user-movie ratings matrix
    user_movie_ratings = ratings_df.pivot_table(index='username', columns='movie_name', values='rating').fillna(0)
    return user_movie_ratings

def fit_svd(user_movie_ratings):
    # Convert to sparse matrix
    ratings_matrix = csr_matrix(user_movie_ratings.values)

    # Apply SVD
    svd = TruncatedSVD(n_components=30) # You can adjust the number of components
    matrix_reduced = svd.fit_transform(ratings_matrix)

    # Compute similarity scores
    user_similarity = cosine_similarity(matrix_reduced)
    return svd, user_similarity

def update_movie_details_df_with_genre_and_actor(movie_details_df):
    # Splitting genres and actors into lists
    movie_details_df['genres'] = movie_details_df['genres'].apply(lambda x: x.split(', ') if x else [])
    movie_details_df['actors'] = movie_details_df['actors'].apply(lambda x: x.split(', ') if x else [])

    # Assuming 'movies_details_df' has columns 'genres' and 'actors' properly formatted as lists of strings
    mlb_genres = MultiLabelBinarizer()
    genres_encoded = mlb_genres.fit_transform(movie_details_df['genres'])

    print('genres encoded')
    # Calculate actor frequency
    actor_counts = movie_details_df['actors'].explode().value_counts()

    # Select top N actors (e.g., top 100 actors)
    top_actors = actor_counts.head(150).index

    # Filter actors data to include only top actors
    filtered_actors = movie_details_df['actors'].apply(lambda x: [actor for actor in x if actor in top_actors])


    mlb_actors = MultiLabelBinarizer()
    actors_encoded = mlb_actors.fit_transform(filtered_actors)
    actor_columns = ['actor_' + col for col in mlb_actors.classes_]
    df_actors_encoded = pd.DataFrame(actors_encoded, columns=actor_columns)
    print('actors encoded')
    # Adding prefixes to the new columns to avoid any overlap
    genre_columns = ['genre_' + col for col in mlb_genres.classes_]
    actor_columns = ['actor_' + col for col in mlb_actors.classes_]

    print('genre columns')
    #print out the genre columns
    print(genre_columns)
    # Creating DataFrames from the encoded arrays
    df_genres_encoded = pd.DataFrame(genres_encoded, columns=genre_columns)
    df_actors_encoded = pd.DataFrame(actors_encoded, columns=actor_columns)

    print('df genres encoded')
    # Joining the new DataFrames back to the original DataFrame
    # Ensuring the index aligns if the DataFrame indexes have been altered
    movie_details_df = movie_details_df.join(df_genres_encoded)
    print('joined genres')
    movie_details_df = movie_details_df.join(df_actors_encoded)
    print('joined actors')
    # Check the updated DataFrame
    print(movie_details_df.head())
    print(movie_details_df.shape)

    return movie_details_df, genre_columns, actor_columns

'''
Generate the SVD vectors for each user, merge with movie details, and prepare the data for training.'''
def get_user_svd_vectors(user_movie_ratings, svd):
    # Generate the SVD vectors for each user
    ratings_matrix = csr_matrix(user_movie_ratings.values)
    user_svd_vectors = svd.transform(ratings_matrix)  # This produces a matrix of shape (n_users, n_components=20)
    print('user_svd_vectors made')
    print(user_svd_vectors.shape)
    # Create a DataFrame for SVD vectors, naming the columns for clarity
    df_user_svd = pd.DataFrame(user_svd_vectors, index=user_movie_ratings.index, columns=[f'svd_{i}' for i in range(user_svd_vectors.shape[1])])
    print('df_user_svd made')
    print(df_user_svd.shape)
    return df_user_svd

def merge_user_movie_details(user_movie_ratings, df_user_svd, movie_details_df, genre_columns, actor_columns):
    # Flatten the user-movie matrix to create a long format DataFrame
    user_movie_long = user_movie_ratings.stack().reset_index()
    user_movie_long.columns = ['username', 'movie_name', 'rating']
    print('user_movie_long made')
    print(user_movie_long.shape)

    # remove and rows with rating 0
    user_movie_long = user_movie_long[user_movie_long['rating'] > 0]
    print('removed rows with rating 0')
    # Merge with SVD vectors
    # Ensure only relevant SVD columns are merged
    # split user_movie_long into train and test
    train_data, user_movie_long = train_test_split(user_movie_long, test_size=0.7, random_state=42)
    user_movie_long = user_movie_long.merge(df_user_svd, on='username', how='left')
    print('merged with svd')

    # Merge with the movie details DataFrame
    # Here, make sure that movie_details_df is prepared and contains only the necessary columns
    # get column list from movie_details_df which contains all genre_columns and actor_columns and movie_name
    columns = ['movie_name'] + genre_columns + actor_columns
    user_movie_details = user_movie_long.merge(movie_details_df[columns], on='movie_name', how='left')
    print('merged with movie details')

    # The DataFrame `user_movie_details` now has the user's SVD vector, movie's genre and actor encoding, and the rating
    # Display the final DataFrame to verify
    print(user_movie_details.head())
    return user_movie_details

def get_data_for_training(user_movie_details):
    # Assume the DataFrame `user_movie_details` includes necessary numeric features
    # Encode categorical features if they're not yet encoded
    # Example for encoding (uncomment if needed):
    # encoder = OneHotEncoder()
    # encoded_features = encoder.fit_transform(user_movie_details[['genres', 'actors']].apply(lambda x: ','.join(x), axis=1))
    print(user_movie_details.columns)
    # remove and row where rating is 0
    print(len(user_movie_details))
    user_movie_details = user_movie_details[user_movie_details['rating'] != 0]
    print(len(user_movie_details))
    # Here, we consider only the SVD features and numeric encoding of genres and actors
    X = user_movie_details.drop(['username', 'movie_name', 'rating'], axis=1)
    y = user_movie_details['rating']

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def train_nn_model(X_train, y_train, X_test, y_test):
    # make nn model
    model = Sequential()
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(128, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='linear'))
    # set learning rate
    model.compile(optimizer='adam', loss='mse')

    # set the learning rate
    model.optimizer.lr = 0.001

    # Train the model
    model.fit(X_train, y_train, verbose=1, epochs = 3)
    return model

def evaluate_nn_model(model, X_test, y_test):



    # Predict on the test set
    y_pred = model.predict(X_test)

    # Evaluate the model using RMSE
    mse = mean_squared_error(y_test, y_pred)
    print(y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Root Mean Squared Error: {rmse}")
    print(f"Mean Absolute Error: {mae}")

def make_movie_name_dict(movie_details_df):

    movie_name_to_real_movie_name = movie_details_df.set_index('movie_name')['real_movie_name'].to_dict()
    # save the dictionary to a file
    import pickle
    with open('movie_name_to_real_movie_name.pkl', 'wb') as f:
        pickle.dump(movie_name_to_real_movie_name, f)

    return movie_name_to_real_movie_name


def main():
    # Connect to the database and load movie details
    ratings_df, movie_details_df = connect_to_db_and_read_and_filter()

    # Split the data into training and testing
    ratings_df, test_data = split_data(ratings_df)

    # Create the user-movie ratings matrix
    user_movie_ratings = create_user_movie_ratings_matrix(ratings_df)

    # Apply SVD
    svd, user_similarity = fit_svd(user_movie_ratings)

    # Update the movie_details_df with genre and actor columns
    movie_details_df, genre_columns, actor_columns = update_movie_details_df_with_genre_and_actor(movie_details_df)

    # Generate the SVD vectors for each user
    df_user_svd = get_user_svd_vectors(user_movie_ratings, svd)

    # Merge user-movie ratings with movie details
    user_movie_details = merge_user_movie_details(user_movie_ratings, df_user_svd, movie_details_df, genre_columns, actor_columns)

    # Prepare the data for training
    X_train, X_test, y_train, y_test = get_data_for_training(user_movie_details)

    # Train the neural network model
    model = train_nn_model(X_train, y_train, X_test, y_test)

    # Evaluate the model
    evaluate_nn_model(model, X_test, y_test)

    # Create a dictionary mapping movie_name to real_movie_name
    movie_name_to_real_movie_name = make_movie_name_dict(movie_details_df)

if __name__ == "__main__":
    main()