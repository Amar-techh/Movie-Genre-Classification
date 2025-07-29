import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.naive_bayes import MultinomialNB # We'll use this later
# from sklearn.metrics import accuracy_score, classification_report # We'll use this later

# --- NLTK Data Downloads (Run once, then you can comment these out) ---
# Uncomment the lines below, run the script once to download, then comment them back.
#nltk.download('punkt')
#nltk.download('stopwords')
#nltk.download('wordnet')
#nltk.download('omw-1.4')
#nltk.download('punkt_tab')
# ---------------------------------------------------------------------

# --- Step 1: Load Data ---
# Define column names based on the file structure (Movie ID, Title, Genre, Description)
def load_data(filepath, columns):
    """
    Loads data from a text file, handling the ':::' delimiter.
    """
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split(' ::: ')
            if len(parts) == len(columns):
                data.append(parts)
            else:
                # print(f"Skipping malformed line in {filepath}: {line.strip()}") # Uncomment for debugging malformed lines
                pass # Silently skip malformed lines for robustness
    return pd.DataFrame(data, columns=columns)

# Define column names for each dataset
train_columns = ['Movie ID', 'Title', 'Genre', 'Description']
test_columns = ['Movie ID', 'Title', 'Description'] # Test data does not have 'Genre'
test_solution_columns = ['Movie ID', 'Title', 'Genre', 'Description']


# Load the datasets
try:
    train_df = load_data('train_data.txt', train_columns)
    test_df = load_data('test_data.txt', test_columns)
    test_solution_df = load_data('test_data_solution.txt', test_solution_columns)

    print("--- Training Data (train_df) Head ---")
    print(train_df.head())
    print("\n--- Test Data (test_df) Head ---")
    print(test_df.head())
    print("\n--- Test Solution Data (test_solution_df) Head ---")
    print(test_solution_df.head())

    print(f"\nTraining data shape: {train_df.shape}")
    print(f"Test data shape: {test_df.shape}")
    print(f"Test solution data shape: {test_solution_df.shape}")

except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the .txt files are in the same directory as your script.")
    # Exit the script if files are not found, as subsequent steps will fail
    exit()
except Exception as e:
    print(f"An unexpected error occurred during data loading: {e}")
    exit()

# --- Step 2: Text Preprocessing ---
# Initialize NLTK components
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Function for text preprocessing
def preprocess_text(text):
    # Ensure text is a string, handle potential NaN values
    if not isinstance(text, str):
        return "" # Return empty string for non-string input

    # 1. Lowercasing
    text = text.lower()

    # 2. Removing Punctuation and Special Characters (keep only alphanumeric and spaces)
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    # 3. Tokenization
    tokens = word_tokenize(text)

    # 4. Removing Stop Words and 5. Lemmatization
    processed_tokens = []
    for word in tokens:
        if word not in stop_words:
            processed_tokens.append(lemmatizer.lemmatize(word))
    
    # Join tokens back into a single string
    return ' '.join(processed_tokens)

print("\n--- Applying Text Preprocessing ---")

# Apply preprocessing to 'Title' and 'Description' columns of training data
train_df['Processed_Text'] = train_df['Title'].fillna('') + ' ' + train_df['Description'].fillna('')
train_df['Processed_Text'] = train_df['Processed_Text'].apply(preprocess_text)

# Apply preprocessing to 'Title' and 'Description' columns of test data
test_df['Processed_Text'] = test_df['Title'].fillna('') + ' ' + test_df['Description'].fillna('')
test_df['Processed_Text'] = test_df['Processed_Text'].apply(preprocess_text)

# Apply preprocessing to 'Title' and 'Description' columns of test solution data (for consistency if needed later)
test_solution_df['Processed_Text'] = test_solution_df['Title'].fillna('') + ' ' + test_solution_df['Description'].fillna('')
test_solution_df['Processed_Text'] = test_solution_df['Processed_Text'].apply(preprocess_text)


print("\n--- Training Data with Processed Text Head ---")
print(train_df[['Movie ID', 'Title', 'Description', 'Processed_Text', 'Genre']].head())

print("\n--- Test Data with Processed Text Head ---")
print(test_df[['Movie ID', 'Title', 'Description', 'Processed_Text']].head())

print("\n--- Test Solution Data with Processed Text Head ---")
print(test_solution_df[['Movie ID', 'Title', 'Description', 'Processed_Text', 'Genre']].head())


# --- Step 3: Feature Extraction (TF-IDF) ---
print("\n--- Performing Feature Extraction (TF-IDF) ---")

# Initialize TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(max_features=5000, min_df=5)

# Fit the vectorizer on the training data's processed text
X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['Processed_Text'])
X_test_tfidf = tfidf_vectorizer.transform(test_df['Processed_Text'])

# For the solution data, we also transform it (useful if we want to use its features later)
X_test_solution_tfidf = tfidf_vectorizer.transform(test_solution_df['Processed_Text'])

# Get the target variable (genres) for training
y_train = train_df['Genre']
y_test_solution = test_solution_df['Genre']


print(f"Shape of TF-IDF features for training data (X_train_tfidf): {X_train_tfidf.shape}")
print(f"Shape of TF-IDF features for test data (X_test_tfidf): {X_test_tfidf.shape}")
print(f"Shape of TF-IDF features for test solution data (X_test_solution_tfidf): {X_test_solution_tfidf.shape}")
print(f"Number of unique genres in training data: {y_train.nunique()}")
print(f"Top 10 genres in training data:\n{y_train.value_counts().head(10)}")

from sklearn.naive_bayes import MultinomialNB
# from sklearn.linear_model import LogisticRegression # Another good option
# from sklearn.svm import LinearSVC # Another good option

print("\n--- Step 4: Model Selection and Training ---")

# Initialize a Multinomial Naive Bayes classifier
classifier = MultinomialNB()

# Train the classifier on the training data (TF-IDF features and genres)
print("Training the Multinomial Naive Bayes model...")
classifier.fit(X_train_tfidf, y_train)
print("Model training complete.")

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import numpy as np # Import numpy for potential future use or if needed by metrics

print("\n--- Step 5: Making Predictions on Test Data ---")

# Make predictions on the preprocessed test data features
y_pred = classifier.predict(X_test_tfidf)

print("Predictions made successfully.")

print("\n--- Step 6: Model Evaluation ---")

# Evaluate the model's performance
# Accuracy Score
accuracy = accuracy_score(y_test_solution, y_pred)
print(f"Accuracy: {accuracy:.4f}")

# Classification Report (provides precision, recall, f1-score for each class)
# To avoid warnings about "too many labels", we can limit to a subset or suppress.
# For now, let's print the full report, but be aware it can be very long.
print("\nClassification Report:")
# Some genres might not be present in predictions if they are rare in test set,
# or might not be predicted. We can handle this by defining target_names
# based on all unique genres or by letting the report handle it.
# For simplicity, let's let the report auto-detect.
print(classification_report(y_test_solution, y_pred, zero_division=0))


# Optional: Display a confusion matrix for some top genres (can be very large for all genres)
# from sklearn.preprocessing import LabelEncoder
#
# # Get top N genres for a more manageable confusion matrix view
# top_n_genres = y_train.value_counts().head(10).index.tolist()
#
# # Filter true and predicted labels for top N genres
# filtered_indices = y_test_solution[y_test_solution.isin(top_n_genres)].index
# y_true_filtered = y_test_solution.loc[filtered_indices]
# y_pred_filtered = pd.Series(y_pred, index=y_test_solution.index).loc[filtered_indices]
#
# print(f"\nConfusion Matrix for Top {len(top_n_genres)} Genres:")
# print(confusion_matrix(y_true_filtered, y_pred_filtered, labels=top_n_genres))


# You can also save predictions to a file if needed
# Create a DataFrame for test predictions
# test_predictions_df = test_df[['Movie ID', 'Title', 'Description']].copy()
# test_predictions_df['Predicted_Genre'] = y_pred
# print("\n--- Sample Test Predictions ---")
# print(test_predictions_df.head())
# test_predictions_df.to_csv('test_predictions.csv', index=False)
# print("\nTest predictions saved to 'test_predictions.csv'")


