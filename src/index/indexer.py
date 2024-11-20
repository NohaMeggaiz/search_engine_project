from sklearn.feature_extraction.text import TfidfVectorizer

def create_index(preprocessed_documents):
    """
    Create a TF-IDF index for the preprocessed documents.
    Returns the TF-IDF matrix and the vectorizer.
    """
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(preprocessed_documents)
    return tfidf_matrix, vectorizer
