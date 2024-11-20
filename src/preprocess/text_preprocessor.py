from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

nltk.download("stopwords")
nltk.download("punkt")

def preprocess_text(text):
    """Preprocess text by tokenizing, lowercasing, and removing stopwords."""
    tokens = word_tokenize(text.lower())  # Tokenize and convert to lowercase
    tokens = [word for word in tokens if word.isalnum()]  # Remove punctuation
    tokens = [word for word in tokens if word not in stopwords.words("english")]  # Remove stopwords
    return " ".join(tokens)
