import sqlite3

def count_unique_users_from_db(db_file):
    """Counts unique usernames from the 'followers' table in a SQLite database.

    Args:
        db_file (str): Path to the SQLite database file.

    Returns:
        int: The count of unique usernames.
    """

    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()

        cursor.execute("SELECT username FROM followers")
        usernames = [row[0] for row in cursor.fetchall()]
        print(len(usernames))
        unique_user_set = set(usernames)  # Create a set for uniqueness
        unique_user_count = len(unique_user_set)

        cursor.execute('CREATE TEMPORARY TABLE unique_followers AS SELECT DISTINCT username FROM followers')
        cursor.execute('DELETE FROM followers')
        cursor.execute('INSERT INTO followers (username) SELECT * FROM unique_followers')
        cursor.execute('DROP TABLE unique_followers')
        conn.commit()
        return unique_user_count

    except sqlite3.Error as e:
        print(f"Database error: {e}")
        return None

    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    database_file = "letterboxd_followers.db"  # Replace with your database file name
    count = count_unique_users_from_db(database_file)

    if count is not None:
        print(f"Total unique users in the database: {count}")
    

    