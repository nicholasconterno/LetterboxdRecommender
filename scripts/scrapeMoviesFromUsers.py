import requests
from bs4 import BeautifulSoup
import sqlite3
import threading
import time
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

semaphore = threading.Semaphore(10)

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

def save_to_db(films, db_file, username):
    """Saves scraped films to a SQLite database in batches."""

    conn = sqlite3.connect(db_file)
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT, 
            movie_name TEXT,    
            rating INTEGER,  
            PRIMARY KEY (username, movie_name)
        )
    """)

    batch_size = 50
    for i in range(0, len(films), batch_size):
        batch = films[i:i+batch_size]
        success = False
        for _ in range(100):  # Retry loop
            try:
                cursor.executemany(
                    "INSERT INTO users (username, movie_name, rating) VALUES (?, ?,?) ",
                    [(username, film[0], film[1]) for film in batch] 
                )
                conn.commit()  # Commit the batch
                success = True
                break
            except sqlite3.OperationalError as e :
                if "database is locked" in str(e):
                    time.sleep(0.2)
                else:
                    raise 
            except sqlite3.IntegrityError as e:
                if "UNIQUE constraint failed" not in str(e):
                    print(f"Error inserting batch of films for user {username}: {e}")
                    
        if not success:
            print(f"Failed to insert batch of films for user {username} after retries")

    conn.close()

def process_user(username,pbar):

    """Processes a single user: scrapes films and saves to the database."""
    with semaphore:
        try:
            films = scrape_letterboxd_films(username)
            save_to_db(films, 'my_letterboxd_data.db', username)
        except Exception as e:
            print(f"Error processing user {username}: {e}")
        finally:
            pbar.update(1)
            
            # print(f"Processed {username}")


        
def main():
    conn = sqlite3.connect('letterboxd_followers.db') 
    cursor = conn.cursor()
    usernames = [row[0] for row in cursor.execute("SELECT username FROM followers")]
    
    
    
    conn.commit()
    conn.close()

    conn = sqlite3.connect('my_letterboxd_data.db')
    cursor = conn.cursor()
# drop the entire table if it exists
    cursor.execute("DROP TABLE IF EXISTS users")
    conn.commit()
    conn.close()


    global count
    count = 0
    for i in range(0, 25000, 5000):
        print(i)
        newusernames = usernames[i:i+5000]
        with ThreadPoolExecutor(max_workers=10) as executor:
            #print num active threads
            with tqdm(total=len(newusernames)) as pbar:
                future_to_user = {executor.submit(process_user, username, pbar): username for username in newusernames}
            
                for future in as_completed(future_to_user):
                    username = future_to_user[future]
                    try:
                        data = future.result()
                    except Exception as exc:
                        print(f"{username} generated an exception: {exc}")
                        
           
            

if __name__ == "__main__":
    time_start = time.time()
    main()
    conn = sqlite3.connect('my_letterboxd_data.db')
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM users")
    print(cursor.fetchone()[0])
    conn.close()
    time_end = time.time()
    print(f"Time taken: {time_end - time_start:.2f} seconds")
