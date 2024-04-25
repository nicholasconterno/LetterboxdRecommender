# Movie Recommendation System: Flick Finder

## Overview

Flick Finder is a Flask-based web application designed to provide personalized movie recommendations to users based on their Letterboxd movie ratings. The system leverages advanced machine learning models including singular value decomposition (SVD) and neural networks to predict user preferences and suggest movies accordingly.

## Features

- **User Profile Based Recommendations**: Users can receive movie suggestions based on their individual profiles created from their Letterboxd account ratings.
- **Dynamic Web Interface**: A responsive and intuitive web interface that allows users to easily interact with the system and retrieve their personalized recommendations.
- **Machine Learning Integration**: The system uses a combination of SVD and neural network models to analyze user preferences and predict potential movie matches.

## Installation

To run the Flick Finder application on your local machine, follow these steps:

### Prerequisites

- Python 3.11 only
- Requirements.txt

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/flick-finder.git
   cd flick-finder
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Initialize the Flask application:**
   ```bash
   python app.py
   ```

The application will start running on `http://localhost:5051`. Open this URL in your web browser to access the Flick Finder application.

## Usage

To use the application:

1. **Navigate to the homepage** at `http://localhost:5051`.
2. **Enter your Letterboxd username** in the input field.
3. **Submit the form** to receive movie recommendations based on your past movie ratings.

## Project Structure

- `app.py`: The Flask application file where routes are defined and the app is run.
- `app_utils.py`: Contains utility functions and the main recommendation logic integrating machine learning models.
- `templates/`: Folder containing HTML templates for rendering the web pages.
- `data/`: Contains data files and machine learning models needed for recommendations (e.g., SVD model, neural network model).

## Technologies Used

- **Flask**: A micro web framework for Python, used to build the web application.
- **BeautifulSoup**: Used for web scraping user data from Letterboxd.
- **TensorFlow/Keras**: For implementing and running the neural network model.
- **Scikit-Learn**: Used for SVD model implementation and other machine learning functionalities.
- **Pandas**: For data manipulation and analysis.
- **NumPy**: For numerical operations.

## Contributors

- Nicholas Conterno
