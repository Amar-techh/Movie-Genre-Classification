# Movie Genre Classification

This project aims to classify movie genres based on their titles and descriptions using machine learning techniques.

## Project Overview

The goal of this project is to build a model that can accurately predict the genre of a movie. This is a multi-class text classification problem where a movie's genre is determined from its textual metadata (title and description).

## Dataset

The datasets used in this project are:
* train_data.txt: Contains movie ID, title, genre, and description for training the model.
* test_data.txt: Contains movie ID, title, and description for making predictions.
* test_data_solution.txt: Contains the actual movie ID, title, genre, and description for the test set, used for evaluating the model's performance.

*Data Format:* Each line in the files is structured with fields separated by ` ::: `.
* *Train data:* ID ::: TITLE ::: GENRE ::: DESCRIPTION
* *Test data:* ID ::: TITLE ::: DESCRIPTION
* *Test solution data:* ID ::: TITLE ::: GENRE ::: DESCRIPTION

*Source:* The dataset is sourced from ftp://ftp.fu-berlin.de/pub/misc/movies/database/.

## Project Structure

* movie_genre_classifier.py: The main Python script containing all the code for data loading, preprocessing, feature extraction, model training, prediction, and evaluation.
* train_data.txt: Training dataset.
* test_data.txt: Test dataset (features only).
* test_data_solution.txt: Test dataset with actual genres (for evaluation).
* description.txt: Describes the dataset format.
* .gitignore: Specifies files and directories that Git should ignore.

## Setup and Installation

1.  *Clone the repository* (if you're getting it from GitHub, otherwise ensure all files are in one folder):
    bash
    git clone [https://github.com/YOUR_USERNAME/Movie-Genre-Classification.git](https://github.com/YOUR_USERNAME/Movie-Genre-Classification.git) # Replace with your repo URL
    cd Movie-Genre-Classification
    

2.  *Create a Virtual Environment (Recommended):*
    bash
    python -m venv venv
    # On Windows
    .\venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    

3.  *Install Required Python Libraries:*
    bash
    pip install pandas nltk scikit-learn
    

4.  *Download NLTK Data:*
    The script movie_genre_classifier.py includes lines to download necessary NLTK data. Run the script once. After the first successful run, you can comment out the nltk.download() lines in movie_genre_classifier.py to prevent re-downloading.

    python
    # Inside movie_genre_classifier.py
    import nltk
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('punkt_tab')
    

## How to Run the Code

1.  Place the train_data.txt, test_data.txt, test_data_solution.txt, and description.txt files in the same directory as your movie_genre_classifier.py script.
2.  Ensure your virtual environment is activated.
3.  Execute the Python script from your terminal:
    bash
    python movie_genre_classifier.py
    
    The script will perform the following steps and print outputs to the console:
    * Data Loading and Initial Exploration
    * Text Preprocessing (Lowercasing, Punctuation Removal, Tokenization, Stop Word Removal, Lemmatization)
    * Feature Extraction (TF-IDF)
    * Model Training (Multinomial Naive Bayes)
    * Prediction on Test Data
    * Model Evaluation (Accuracy and Classification Report)

## Model and Performance

* *Model Used*: Multinomial Naive Bayes
* *Feature Extraction*: TF-IDF (Term Frequency-Inverse Document Frequency)
* *Achieved Accuracy*: ~52.24% (on the test dataset)

This accuracy serves as a solid baseline for the movie genre classification task.

## Potential Improvements

To enhance the model's performance, consider exploring:

* *Different Classification Models*:
    * Logistic Regression
    * Support Vector Machines (LinearSVC)
    * Gradient Boosting Classifiers (e.g., LightGBM, XGBoost)
    * Deep Learning models (e.g., LSTMs, GRUs, or pre-trained transformers like BERT for more complex semantic understanding)
* *Feature Engineering*:
    * Experiment with TF-IDF parameters (e.g., ngram_range for bigrams/trigrams).
    * Word Embeddings (Word2Vec, GloVe) or FastText for capturing semantic relationships between words.
* *Hyperparameter Tuning*: Optimize the hyperparameters of the chosen model and the TF-IDF vectorizer using techniques like GridSearchCV or RandomizedSearchCV.
* *Handling Imbalanced Classes*: Investigate techniques for dealing with imbalanced genre distributions (e.g., SMOTE, class weights).
* *Ensemble Methods*: Combine multiple models to potentially achieve better performance.
* *Error Analysis*: Manually examine misclassified examples to understand common failure patterns and identify areas for improvement in preprocessing or features.
