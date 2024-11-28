from django.conf import settings
from django.shortcuts import render
from src.extractors.base_extractor import extract_text
from src.preprocess.text_preprocessor import preprocess_text
from src.index.indexer import create_index
from src.search.search_engine import search
import os

# Load and index the documents at startup
doc_dir = "C:\\Users\\hp\\Documents\\search_engine_project\\data"
documents = extract_text(doc_dir)
preprocessed_documents = [preprocess_text(doc) for doc in documents]
tfidf_matrix, vectorizer = create_index(preprocessed_documents)

def search_view(request):
    query = request.GET.get('q', '')
    results = []

    if query:
        processed_query = preprocess_text(query)
        search_results = search(processed_query, tfidf_matrix, vectorizer)

        for doc_id, score in search_results:
            file_name = os.listdir(doc_dir)[doc_id]
            results.append({
                'score': score,
                'document': documents[doc_id],
                'file_path': f"{settings.DATA_URL}{file_name}"  # Generate data URL
            })

    return render(request, 'search_app/search.html', {'query': query, 'results': results})
