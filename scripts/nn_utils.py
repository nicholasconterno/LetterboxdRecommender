import requests
from bs4 import BeautifulSoup
import pandas as pd
import pickle
import numpy as np
from tensorflow.keras.models import load_model
import time

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
    films = scrape_letterboxd_films(username, start_page=1)
    return films

def load_models():
    # load the svd model
    with open('data/svd.pkl', 'rb') as f:
        svd = pickle.load(f)

#load movie_details_df with pkl

    movie_details_df = pd.read_pickle('data/movie_details_df.pkl')
    model = load_model('data/nn_model.keras')
    with open('data/movie_name_to_real_movie_name.pkl', 'rb') as f:
        movie_name_to_real_movie_name = pickle.load(f)
    return svd, movie_details_df, model, movie_name_to_real_movie_name

def get_user_vector(films, movie_details_df, svd):
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
    user_vector = svd.transform(np.array(user_vector).reshape(1, -1))

    return user_vector, filmsSeen
def get_recommendations(user_vector, movie_details_df, model, movie_name_to_real_movie_name, filmsSeen):
    # merge the user_vector with the movie_details_df
    df_user_vector = pd.DataFrame(user_vector, columns=[f'svd_{i}' for i in range(user_vector.shape[1])])
    print('df_user_vector made')

    # get recommended movies
    user_features = np.tile(user_vector, (len(movie_details_df), 1))

    X_pred = np.hstack([user_features, movie_details_df.drop(['movie_name', 'real_movie_name', 'director', 'genres', 'actors'], axis=1).values])
    y_pred = model.predict(X_pred)

    

    # Get the top_k movies with the highest predicted ratings
    top_indices = np.argsort(-y_pred.flatten())[:30]
    top_movies = movie_details_df.iloc[top_indices]['movie_name'].values
    top_ratings = y_pred.flatten()[top_indices]
    # remove movies with genre 'Documentary' or 'Short' from the recommendations
    
    top_ratings = [top_ratings[i] for i in range(len(top_movies)) if 'Documentary' not in movie_details_df[movie_details_df['movie_name'] == top_movies[i]]['genres'].values[0] and 'Short' not in movie_details_df[movie_details_df['movie_name'] == top_movies[i]]['genres'].values[0]]
    # remove movies with genre 'TV Movie' or 'Music' from the recommendations
    top_movies = [top_movies[i] for i in range(len(top_movies)) if 'Documentary' not in movie_details_df[movie_details_df['movie_name'] == top_movies[i]]['genres'].values[0] and 'Short' not in movie_details_df[movie_details_df['movie_name'] == top_movies[i]]['genres'].values[0]]
    
    # remove movies with genre 'TV Movie' or 'Music' from the recommendations
    top_ratings = [top_ratings[i] for i in range(len(top_movies)) if 'TV Movie' not in movie_details_df[movie_details_df['movie_name'] == top_movies[i]]['genres'].values[0] and 'Music' not in movie_details_df[movie_details_df['movie_name'] == top_movies[i]]['genres'].values[0]]
    top_movies = [top_movies[i] for i in range(len(top_movies)) if 'TV Movie' not in movie_details_df[movie_details_df['movie_name'] == top_movies[i]]['genres'].values[0] and 'Music' not in movie_details_df[movie_details_df['movie_name'] == top_movies[i]]['genres'].values[0]]

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
    
    top_ratings = [round(rating, 2) for rating in top_ratings]
    # divide all ratings by 2
    top_ratings = [rating/2 for rating in top_ratings]

    # print out the top movies and ratings side by side
    for i in range(len(top_movies)):
        print(f"{top_movies_real_names[i]}: {top_ratings[i]}")
    return top_movies_real_names, top_ratings

def main(username):
    # load the models
    svd, movie_details_df, model, movie_name_to_real_movie_name = load_models()
    # scrape films
    films = get_films(username)
    # get user vector
    user_vector, filmsSeen = get_user_vector(films, movie_details_df, svd)
    # get recommendations
    top_movies, top_ratings = get_recommendations(user_vector, movie_details_df, model, movie_name_to_real_movie_name, filmsSeen)
    return top_movies, top_ratings

if __name__ == "__main__":
    t = time.time()
    main('marcoslammel')
    print(time.time()-t)
