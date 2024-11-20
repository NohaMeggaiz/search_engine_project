from sklearn.metrics.pairwise import cosine_similarity

def search(query, tfidf_matrix, vectorizer):
    """
    Search for a query in the TF-IDF index.
    Returns a list of matching document indices and their scores.
    """
    # Vectorize the query
    query_vec = vectorizer.transform([query])
    
    # Calculate cosine similarity
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    
    # Rank documents by similarity
    ranked_indices = similarities.argsort()[::-1]  # Sort in descending order
    results = [(index, similarities[index]) for index in ranked_indices if similarities[index] > 0]
    return results
