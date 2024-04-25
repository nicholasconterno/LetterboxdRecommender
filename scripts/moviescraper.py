import sqlite3
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import time
from requests.exceptions import ConnectionError, Timeout
import backoff

api_key = "c07c9d067d630faff2eed10b673199a3"

@backoff.on_exception(backoff.expo, (ConnectionError, Timeout), max_tries=25)
def get_film_info(title, year=None):
    '''
    Get film info from The Movie Database API'''

    # search_response = requests.get(search_url, headers=headers)
    original_title = title
    # if title is in the format film-title-year, remove the year by checking if the last 4 characters are digits
    if title[-4:].isdigit():
        title = title[:-5]

    # if the title is in the format film-title-year-1, remove the year by checking if the last 5 characters are digits
    if title[-1:].isdigit() and title[-6:-2].isdigit():
        title = title[:-7]
        
    search_url = f"https://api.themoviedb.org/3/search/movie?api_key={api_key}&query={title}"
    if year:
        search_url += f"&year={year}"
    search_response = requests.get(search_url)
    search_data = search_response.json()
    # print(title)
    if search_data['results']:
        first_result = search_data['results'][0]
        film_id = first_result['id']

        # Fetch additional details (e.g., credits for director and actors)
        details_url = f"https://api.themoviedb.org/3/movie/{film_id}?api_key={api_key}&append_to_response=credits"
        details_response = requests.get(details_url)
        details_data = details_response.json()

        directors = [crew['name'] for crew in details_data['credits']['crew'] if crew['job'] == 'Director']
        actors = [actor['name'] for actor in details_data['credits']['cast'][0:20]]  # Top 20 actors
        screenwriters = [crew['name'] for crew in details_data['credits']['crew'] if crew['job'] in ['Screenplay', 'Writer']]
        genres = [genre['name'] for genre in details_data['genres']]
        film_info = {
            'letterboxd_slug': original_title,  # This is the unique identifier for the film, e.g., 'the-godfather
            'title': first_result['title'],
            'release_date': first_result.get('release_date', 'N/A'),
            'overview': first_result.get('overview', 'No overview available.'),
            'vote_average': first_result.get('vote_average', 'N/A'),
            'director': ', '.join(directors),
            'actors': ', '.join(actors),
            'screenwriters': ', '.join(screenwriters),
            'box_office': details_data.get('revenue', 'N/A'),  # Box office data if available
            'genres' : ', '.join(genres)
        }
    else:
        film_info = None


    return film_info

def update_database(film_info):
    conn = sqlite3.connect('my_letterboxd_data.db')
    cursor = conn.cursor()
    # time.sleep(0.1)
    cursor.execute("""
    INSERT INTO film_details_small (letterboxd_slug, movie_name, release_date, overview, director, actors, screenwriters, box_office, genres)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    ON CONFLICT(letterboxd_slug) DO UPDATE SET
    release_date=excluded.release_date,
    overview=excluded.overview,
    director=excluded.director,
    actors=excluded.actors,
    screenwriters=excluded.screenwriters,
    box_office=excluded.box_office,
    genres=excluded.genres
    """, (film_info['letterboxd_slug'],film_info['title'], film_info['release_date'], film_info['overview'], 
          film_info['director'], film_info['actors'], film_info['screenwriters'], film_info['box_office'], film_info['genres']))
    conn.commit()
    conn.close()

def main():
    conn = sqlite3.connect('my_letterboxd_data.db')
    cursor = conn.cursor()

    cursor.execute("SELECT DISTINCT movie_name, year FROM release_years")
    films = cursor.fetchall()

    # dump films to a file
    with open('films123.txt', 'w') as f:
        for film in films:
            f.write(f"{film}\n")

    # read in films from movie_names.txt
    with open('movie_names.txt', 'r') as f:
        filmsNeeded = [line.strip() for line in f.readlines()]
        # print(filmsNeeded)
    print(filmsNeeded)
    finalfilms = []
    for film in tqdm(filmsNeeded):
        for film2 in films:
            dbmoviename = film2[0]
            if film == dbmoviename:
                finalfilms.append(film2)

    films = finalfilms

    
    # cursor.execute("DROP TABLE IF EXISTS film_details")
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS film_details_small (
    letterboxd_slug TEXT PRIMARY KEY,
    movie_name TEXT,
    release_date TEXT,
    overview TEXT,
    director TEXT,
    actors TEXT,
    screenwriters TEXT,
    box_office TEXT,
    genres TEXT       
    )
    """)

    # Load processed films
    with open('films_processed.txt', 'r') as f:
        processed_films = {line.strip() for line in f.readlines()}

    print(f"Processing {len(films)} films")
    with ThreadPoolExecutor(max_workers=1) as executor:
        # Prepare tasks, skipping already processed films
        tasks = {
            executor.submit(get_film_info, film[0].strip('/'), film[1]): film 
            for film in films
            if film[0].strip('/') not in processed_films
        }

        for future in tqdm(as_completed(tasks), total=len(tasks)):
            try:
                film_info = future.result()
                print(film_info)
                if film_info:
                    update_database(film_info)
                    with open('films_processed.txt', 'a') as f:
                        # Write film slug to file to mark it as processed
                        f.write(f"{film_info['letterboxd_slug']}\n")
                else:
                    with open('films_not_found.txt', 'a') as f:
                        # Write film slug to file to mark it as not found
                        f.write(f"{tasks[future][0].strip('/')}\n")
            except Exception as e:
                print(f"Error processing film: {tasks[future]} - {str(e)}")

if __name__ == "__main__":
    #print how many items are in the database
    conn = sqlite3.connect('my_letterboxd_data.db')
    cursor = conn.cursor()
    main()

    cursor.execute("SELECT COUNT(*) FROM film_details_small")
    print(cursor.fetchone()[0])
    # commit
    conn.commit()


