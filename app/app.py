from flask import Flask, render_template, request, jsonify
import json
from app_utils import main  # Ensure the main function is correctly imported

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    movies_posters = []  # Initialize empty list for movies and posters
    error = None  # Initialize no error

    if request.method == 'POST':
        username = request.form['username']
        if username:
            try:
                # Assuming main function is imported from your script and returns (real_movies, ratings, movies)
                real_movies, ratings, movies = main(username)
                with open('data/slug_to_poster.json', 'r') as file:
                    slug_to_poster = json.load(file)
                movies_posters = [(real_movies[idx], ratings[idx], slug_to_poster.get(movie, 'default_poster.jpg'))
                                  for idx, movie in enumerate(movies)]
            except Exception as e:
                error = str(e)  # Capture the error to display on the page

    # Render the same template whether GET or POST, but with different data
    return render_template('index.html', movies_posters=movies_posters, error=error)

if __name__ == '__main__':
    app.run(debug=True)
