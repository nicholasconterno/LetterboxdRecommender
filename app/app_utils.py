import requests
from bs4 import BeautifulSoup
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import time
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import load_npz

SCORE_DICT = {
    "½": 1,
    "★": 2,
    "★½": 3,
    "★★": 4,
    "★★½": 5,
    "★★★": 6,
    "★★★½": 7,
    "★★★★": 8,
    "★★★★½": 9,
    "★★★★★": 10,
}


def scrape_letterboxd_films(username, start_page=1, max_pages=None):
    """Scrapes film names from a Letterboxd user's 'films watched' pages.

    Args:
        username (str): The Letterboxd username.
        start_page (int): The page number to start scraping from. Defaults to 1.
        max_pages (int): The maximum number of pages to scrape. Defaults to None (scrape all).
    """

    films = []
    page_num = start_page

    while True:
        url = f"https://letterboxd.com/{username}/films/page/{page_num}/"
        response = requests.get(url)
        response.raise_for_status()

        soup = BeautifulSoup(response.content, 'html.parser')

        new_films=[]

        #find all the film elements, they are always after 'data-film-name'
        for film_element in soup.find_all('li', class_='poster-container'):
            title = film_element.find('div', class_='film-poster').get('data-film-slug')
            rating_element = film_element.find('span', class_='rating')

            if rating_element:
                rating = rating_element.text.strip()
                rating = SCORE_DICT.get(rating, None)
            else:
                rating = None
            new_films.append((title, rating))
            films.append((title, rating))


        page_num += 1
        if len(new_films) == 0 or (max_pages and page_num > max_pages):
            break

    return films

def get_films(username):
    '''
    Get the films watched by a Letterboxd user'''
    films = scrape_letterboxd_films(username, start_page=1)
    return films

def load_models():
    '''
    Load the models and data needed for the recommendation system'''
    # load the svd model
    with open('data/svd.pkl', 'rb') as f:
        svd = pickle.load(f)

#load movie_details_df with pkl

    movie_details_df = pd.read_pickle('data/movie_details_df.pkl')
    model = load_model('data/nn_model.keras')
    with open('data/movie_name_to_real_movie_name.pkl', 'rb') as f:
        movie_name_to_real_movie_name = pickle.load(f)
    
    # load svd model
    with open('data/svd_model.pkl', 'rb') as f:
        svd_model = pickle.load(f)
    return svd, movie_details_df, model, movie_name_to_real_movie_name, svd_model

def get_user_vector(films, movie_details_df, svd, svd_model):
    '''
    Get the user vector for the user'''
    # create user_vector for the user
    user_vector = [0] * len(movie_details_df)
    filmsSeen = []
    for film in films:
        filmsSeen.append(film[0])
        if film[0] in movie_details_df['movie_name'].values:
            movie_index = movie_details_df[movie_details_df['movie_name'] == film[0]].index[0]
            if film[1] != None:
                user_vector[movie_index] = film[1]

    # use svd to transform the user_vector
    user_vector_t = svd.transform(np.array(user_vector).reshape(1, -1))

    user_vector_for_svd = svd_model.transform(csr_matrix(user_vector).reshape(1, -1))

    user_vector = user_vector_t
    return user_vector, filmsSeen, user_vector_for_svd
def get_recommendations(user_vector, movie_details_df, model, movie_name_to_real_movie_name, filmsSeen):
    '''
    Get movie recommendations for the user'''
    # merge the user_vector with the movie_details_df
    df_user_vector = pd.DataFrame(user_vector, columns=[f'svd_{i}' for i in range(user_vector.shape[1])])
    # print('df_user_vector made')

    # drop columns with genres of 'Documentary' or 'Short' or 'TV Movie' or 'Music'
    # movie_details_df = movie_details_df[~movie_details_df['genre_Documentary']==1]
    # # movie_details_df = movie_details_df[~movie_details_df['genre_Short']==1]
    # movie_details_df = movie_details_df[~movie_details_df['genre_TV Movie']==1]
    # movie_details_df = movie_details_df[~movie_details_df['genre_Music']==1]

    # get recommended movies
    user_features = np.tile(user_vector, (len(movie_details_df), 1))

    X_pred = np.hstack([user_features, movie_details_df.drop(['movie_name', 'real_movie_name', 'director', 'genres', 'actors'], axis=1).values])
    y_pred = model.predict(X_pred)

    

    # Get the top_k movies with the highest predicted ratings
    top_indices = np.argsort(-y_pred.flatten())[:100]
    top_movies = movie_details_df.iloc[top_indices]['movie_name'].values
    top_ratings = y_pred.flatten()[top_indices]
    # remove movies with genre 'Documentary' or 'Short' from the recommendations
    
    # top_ratings = [top_ratings[i] for i in range(len(top_movies)) if 'Documentary' not in movie_details_df[movie_details_df['movie_name'] == top_movies[i]]['genres'].values[0] and 'Short' not in movie_details_df[movie_details_df['movie_name'] == top_movies[i]]['genres'].values[0]]
    # # remove movies with genre 'TV Movie' or 'Music' from the recommendations
    # top_movies = [top_movies[i] for i in range(len(top_movies)) if 'Documentary' not in movie_details_df[movie_details_df['movie_name'] == top_movies[i]]['genres'].values[0] and 'Short' not in movie_details_df[movie_details_df['movie_name'] == top_movies[i]]['genres'].values[0]]
    
    # # remove movies with genre 'TV Movie' or 'Music' from the recommendations
    # top_ratings = [top_ratings[i] for i in range(len(top_movies)) if 'TV Movie' not in movie_details_df[movie_details_df['movie_name'] == top_movies[i]]['genres'].values[0] and 'Music' not in movie_details_df[movie_details_df['movie_name'] == top_movies[i]]['genres'].values[0]]
    # top_movies = [top_movies[i] for i in range(len(top_movies)) if 'TV Movie' not in movie_details_df[movie_details_df['movie_name'] == top_movies[i]]['genres'].values[0] and 'Music' not in movie_details_df[movie_details_df['movie_name'] == top_movies[i]]['genres'].values[0]]

    top_ratings = [top_ratings[i] for i in range(len(top_movies)) if top_movies[i] not in filmsSeen]
    top_movies = [top_movies[i] for i in range(len(top_movies)) if top_movies[i] not in filmsSeen]

    #remove cider house rules
    top_ratings = [top_ratings[i] for i in range(len(top_movies)) if top_movies[i] != 'the-cider-house-rules']
    top_movies = [top_movies[i] for i in range(len(top_movies)) if top_movies[i] != 'the-cider-house-rules']
    # load the movie_name_to_real_movie_name dictionary

    top_movies_real_names = [movie_name_to_real_movie_name[movie] for movie in top_movies]
    # get highest rating
    highest_rating = max(top_ratings)
    if highest_rating>=10:
        # if the highest rating is 10, divide all ratings by 2
        top_ratings = [rating/((highest_rating/10)+.01) for rating in top_ratings]
    
    
    # divide all ratings by 2
    top_ratings = [rating/2 for rating in top_ratings]
    top_ratings = [round(rating, 1) for rating in top_ratings]

    # print out the top movies and ratings side by side
    # for i in range(len(top_movies)):
    #     print(f"{top_movies_real_names[i]}: {top_ratings[i]}")
    return top_movies_real_names[0:10], top_ratings[0:10], top_movies[0:10]

def load_data_for_SVD():
    '''
    Load the data needed for the SVD model'''
    # load with npz the ratings matrix
    # data/ratings_matrix.npz
    ratings_matrix = load_npz('data/ratings_matrix.npz')
    with open('data/movie_index.pkl', 'rb') as f:
        movie_index = pickle.load(f)
    with open('data/reduced_matrix.pkl', 'rb') as f:
        matrix_reduced = pickle.load(f)
    with open('data/movie_details_df.pkl', 'rb') as f:
        movie_details_df = pickle.load(f)
    return ratings_matrix, movie_index, matrix_reduced, movie_details_df

def predict_movies_for_new_user(new_user_ratings,svd_model, movie_index,matrix_reduced,ratings_matrix,movie_details_df, top_k=12):
    '''
    Predict movies for a new user based on the ratings given by the user'''
    # Create a new user ratings vector with the same columns as the original ratings matrix
    # load in movie_name_to_real_movie_name
    with open('data/movie_name_to_real_movie_name.pkl', 'rb') as f:
        movie_name_to_real_movie_name = pickle.load(f)
    
    films_seen = set(movie_name_to_real_movie_name.get(movie, '') for movie, rating in new_user_ratings.items() if rating != 0)

    movie_ids = list(movie_index.keys())  # movie_index should map movie names to column indices
    new_user_vector = np.zeros(len(movie_ids))
    # for movie, rating in new_user_ratings.items():
    #     if movie in movie_index:
    #         new_user_vector[movie_index[movie]] = rating
    indices = [movie_index[movie] for movie, rating in new_user_ratings.items() if rating != 0 and movie in movie_index]
    ratings = [rating for movie, rating in new_user_ratings.items() if rating != 0 and movie in movie_index]
    new_user_vector[indices] = ratings

    # Transform new user vector using the existing SVD model and compute similarity
    new_user_vector = svd_model.transform(csr_matrix(new_user_vector).reshape(1, -1))
    new_user_similarity = cosine_similarity(new_user_vector, matrix_reduced).flatten()

    # Exclude the new user's self-comparison and get indices of top similar users
    top_users_indices = np.argsort(-new_user_similarity)[1:25]
    top_users_ratings = ratings_matrix[top_users_indices]

    # Convert to dense and filter movies rated by at least 5 top users
    dense_ratings = top_users_ratings.toarray()
    valid_movies = (dense_ratings > 0).sum(axis=0) >= 5
    top_users_ratings = dense_ratings[:, valid_movies]
    
    # print(len(valid_movies))
    # Calculate the mean of ratings, ignoring zeros
    recommended_movies = np.apply_along_axis(lambda x: np.mean(x[x > 0]), 0, top_users_ratings)
    # print(len(recommended_movies))
    # print(recommended_movies)
    # Get the names of valid movies
    valid_movie_names = np.array(movie_ids)[valid_movies]
    # print(valid_movie_names)
    with open('data/movie_name_to_real_movie_name.pkl', 'rb') as f:
        movie_name_to_real_movie_name = pickle.load(f)
    new_valid_movies_names = []
    for i in range(len(valid_movie_names)):
        if valid_movie_names[i] in movie_name_to_real_movie_name.keys():
            new_valid_movies_names.append( movie_name_to_real_movie_name[valid_movie_names[i]])
    valid_movie_names = new_valid_movies_names
    
    recommended_movies = dict(zip(valid_movie_names, recommended_movies))

    
    

    # Filter based on movie details and documentaries
    recommended_movies = {movie: rating for movie, rating in recommended_movies.items() if movie in movie_details_df['real_movie_name'].values}
    recommended_movies = {movie: rating for movie, rating in recommended_movies.items() if 'Documentary' not in movie_details_df.set_index('real_movie_name').loc[movie, 'genres']}
    recommended_movies = {movie: rating for movie, rating in recommended_movies.items() if 'Music' not in movie_details_df.set_index('real_movie_name').loc[movie, 'genres']}
    recommended_movies = {movie: rating for movie, rating in recommended_movies.items() if 'TV Movie' not in movie_details_df.set_index('real_movie_name').loc[movie, 'genres']}

    # Remove movies the user has already rated
    recommended_movies = {movie: rating for movie, rating in recommended_movies.items() if movie not in  films_seen}
    # Sort and select the top_k movies
    recommended_movies = sorted(recommended_movies.items(), key=lambda x: x[1], reverse=True)[:top_k]
    # return 3 lists, one with the real movie names, one with the ratings and one with the movie names (use the map)
    recommended_movies_real = [movie[0] for movie in recommended_movies]
    recommended_movies_ratings = [movie[1] for movie in recommended_movies]
    # load in real_movie_name_to_movie_name
    with open('data/real_movie_name_to_movie_name.pkl', 'rb') as f:
        real_movie_name_to_movie_name = pickle.load(f)
    recommended_movies = [real_movie_name_to_movie_name[movie[0]] for movie in recommended_movies]
    return recommended_movies_real, recommended_movies_ratings, recommended_movies



def main(username):
    '''
    Main function to get recommendations for a user'''
    # load the models
    t = time.time()
    svd, movie_details_df, model, movie_name_to_real_movie_name, svd_model = load_models()
    # scrape films
    films = get_films(username)
    # get user vector
    user_vector, filmsSeen, user_vector_for_svd = get_user_vector(films, movie_details_df, svd, svd_model)
    # get recommendations

    top_movies_real, top_ratings, top_movies = get_recommendations(user_vector, movie_details_df, model, movie_name_to_real_movie_name, filmsSeen)
    
    # return the top movies and ratings
    # load data for SVD
    ratings_matrix, movie_index, matrix_reduced, movie_details_df = load_data_for_SVD()
    template = pd.read_pickle('data/ratings_template.pkl')
    # loop through films and add to template
    films_seen = []
    for film in films:
        template.loc[username, film[0]] = film[1]
        
    # predict movies for new user
    new_user_ratings = template.loc[username]
    # fill missing values with 0
    new_user_ratings = new_user_ratings.fillna(0)
    rec_movies_real, rec_ratings, rec_movies = predict_movies_for_new_user(new_user_ratings,svd_model, movie_index,matrix_reduced,ratings_matrix,movie_details_df)
    # round rec_ratings to 2 decimal places make sure 2 decimal places
    rec_ratings = [round(rating, 1) for rating in rec_ratings]
    for i in range(len(rec_movies_real)):
        print(f"{rec_movies_real[i]}: {rec_ratings[i]}")
    print('Time taken: ', time.time()-t)

    # combine topmovies and rec_movies sorted by rating
    rec_movies_real = rec_movies_real + top_movies_real
    rec_ratings = rec_ratings + top_ratings
    rec_movies = rec_movies + top_movies
    # sort by rating
    rec_movies_real = [x for _, x in sorted(zip(rec_ratings, rec_movies_real), reverse=True)]
    rec_movies = [x for _, x in sorted(zip(rec_ratings, rec_movies), reverse=True)]
    rec_ratings = sorted(rec_ratings, reverse=True)
    for i in range(len(rec_movies_real)):
        print(f"{rec_movies_real[i]}: {rec_ratings[i]}")
    
    return rec_movies_real[0:12], rec_ratings[0:12], rec_movies[0:12]

if __name__ == "__main__":
    recommendations = main('nconterno')
    # print(recommendations)