import requests
from bs4 import BeautifulSoup
import time 
import sqlite3
from threading import Thread, Lock
import os
from queue import Queue

base_url = "https://letterboxd.com/"
initial_user = "nconterno"
max_depth = 10

visited_users = set()
task_queue = Queue()  # Queue for tasks
task_queue.put((initial_user, 0))  # Add the initial user to the queue
db_lock = Lock() # Lock for database access

def scrape_followers(username, current_depth=0, page_num=1):
    global visited_users
    global to_scrape
    if len(visited_users) > 1000000:
        return
    if current_depth > max_depth:
        return

    followers_url = base_url + username + "/followers/page/" + str(page_num) + "/"
    response = requests.get(followers_url)
    response.raise_for_status()  
    # print(username)
    soup = BeautifulSoup(response.content, "html.parser")
    follower_rows = soup.find_all("tr")

    found_user = False

    for row in follower_rows:
        try:
            username_element = row.find("td", class_="table-person").find("a", class_="name")
            if username_element:
                new_username = username_element.get('href').strip('/')
                if new_username not in visited_users:
                    visited_users.add(new_username)
                    save_to_db(new_username, current_depth)
                    task_queue.put((new_username, current_depth + 1))
                    if len(visited_users) % 50 == 0:
                        print(f"Found {new_username} at depth {current_depth} (Total: {len(visited_users)})")
                    found_user = True
        except AttributeError:
            pass

    if not found_user:
        return

    #  Scrape the next page of the same user (maintain depth)
    scrape_followers(username, current_depth, page_num + 1)  

def save_to_db(username, depth):
    conn = sqlite3.connect('letterboxd_followers.db')  # Create connection inside save_to_db
    cursor = conn.cursor()
    cursor.execute("INSERT INTO followers (username, depth) VALUES (?, ?)", (username, depth))
    conn.commit()  
    conn.close() 

def worker():
    while True:  
        username, current_depth = task_queue.get()  
        try:
            scrape_followers(username, current_depth) 
        except Exception as e:  # Catch any unhandled exceptions
            print(f"Unexpected error for {username} - {e}. Skipping...")
        finally:
            task_queue.task_done() 

if __name__ == "__main__":    
    # Database setup 
    conn = sqlite3.connect('letterboxd_followers.db')
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE IF NOT EXISTS followers (username TEXT, depth INTEGER)")

    thread_count = 30  # Choose your desired thread count 
    print(f"\n*** Testing with {thread_count} threads ***")


    visited_users = set()
    to_scrape = [(initial_user, 0)]

    start_time = time.time()  
    thread_start_time = time.time()

    threads = []
    for _ in range(thread_count):
        t = Thread(target=worker)
        t.start()
        threads.append(t)

    for t in threads:
        t.join()  

    task_queue.join()  # Wait for the queue to be empty
    thread_end_time = time.time()
    scraping_start_time = time.time()  
    scrape_followers(initial_user)  
    scraping_end_time = time.time()

    thread_total_time = thread_end_time - thread_start_time  
    scraping_time = scraping_end_time - scraping_start_time

    print(f"Finished with {thread_count} threads.")
    print(f"Thread startup time: {thread_total_time:.2f} seconds")
    print(f"Scraping time: {scraping_time:.2f} seconds")

    # Save the database
    conn.commit() 
    conn.close() 
