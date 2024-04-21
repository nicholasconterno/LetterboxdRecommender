import sqlite3
from bs4 import BeautifulSoup
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

# Function to scrape the release year of a single film
def scrape_film_year(film_slug):
    try:
        # Scrape the film page
        html = requests.get(f"https://letterboxd.com/film/{film_slug}/")
        soup = BeautifulSoup(html.content, 'html.parser')
        title = soup.find('title')
        # Assuming the year is always formatted correctly in the title
        year=None
        # assume format is "Film Title directed by Director (Year) so get the year from the title"
        if 'directed' in title.text:
            year = title.text.split('directed')[0].strip()[-5:-1]
            year = int(year)  # Convert the year to an integer
        # assume format is "Film Title (Year) so get the year from the title"
        elif '(' in title.text and ')' in title.text:
            year = (title.text.split('(')[-1].split(')')[0])
            try:
                year = int(year)  # Convert the year to an integer
            except:
                year = None
        return (film_slug, year)
    except Exception as e:
        print(f"Error scraping {film_slug}: {e}")
        return (film_slug, None)

# Function to insert film and year into the database
def insert_film_year(conn, film_year_tuples):
    cursor = conn.cursor()
    cursor.executemany("INSERT INTO release_years (movie_name, year) VALUES (?, ?)", film_year_tuples)
    conn.commit()

# Main function to scrape years using multithreading
def scrape_and_save_film_years(films):
    # Connect to the database
    conn = sqlite3.connect('my_letterboxd_data.db')
    cursor = conn.cursor()
    
    # Recreate the table to start fresh
    cursor.execute("DROP TABLE IF EXISTS release_years")
    cursor.execute("CREATE TABLE release_years (movie_name TEXT, year INTEGER)")
    
    # Using ThreadPoolExecutor to scrape in parallel
    with ThreadPoolExecutor(max_workers=20) as executor:
        future_to_film = {executor.submit(scrape_film_year, film): film for film in films}
        # Use tqdm to show progress
        for future in tqdm(as_completed(future_to_film), total=len(films), desc="Scraping films"):
            # Get the film slug from the future
            film_slug = future_to_film[future]
            try:
                # Get the result of the future
                film, year = future.result()
                if year is not None:
                    insert_film_year(conn, [(film, year)])
            except Exception as e:
                print(f"Error processing {film_slug}: {e}")

    conn.close()

if __name__ == "__main__":
    # Load film slugs from file
    films = []
    with open('all_films.txt', 'r') as f:
        films = f.read().splitlines()
    # Call the main function
    scrape_and_save_film_years(films)
