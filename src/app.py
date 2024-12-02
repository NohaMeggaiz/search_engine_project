from extractors.base_extractor import extract_text
from preprocess.text_preprocessor import preprocess_text
from index.indexer import create_index
from search.search_engine import search
import nltk

# Ensure necessary NLTK data is available
nltk.download('punkt')
nltk.download('stopwords')

def display_inverted_index(inverted_index):
    """Display the inverted index in a readable format."""
    print("\nInverted Index:")
    for term, postings in inverted_index.items():
        print(f"Term: {term}")
        for doc, score in postings.items():
            print(f"  Document: {doc}, Score: {score}")
    print("\n")

def main():
    # Directory containing documents
    doc_dir = "C:\\Users\\hp\\Documents\\search_engine_project\\data"
    
    # Step 1: Extract text from documents
    print("Extracting text from documents...")
    documents = extract_text(doc_dir)
    print(f"Extracted text from {len(documents)} documents.")
    
    # Step 2: Preprocess the text
    print("Preprocessing documents...")
    preprocessed_documents = [preprocess_text(doc) for doc in documents]
    
    # Step 3: Create the index
    print("Creating TF-IDF index...")
    tfidf_matrix, vectorizer = create_index(preprocessed_documents)
    
    # Build the inverted index from the TF-IDF data
    feature_names = vectorizer.get_feature_names_out()
    inverted_index = {
        term: {
            f"Document {doc_id}": tfidf_matrix[doc_id, term_id]
            for doc_id in range(tfidf_matrix.shape[0])
            if tfidf_matrix[doc_id, term_id] > 0
        }
        for term_id, term in enumerate(feature_names)
    }
    
    # Display the inverted index
    display_inverted_index(inverted_index)
    
    # Step 4: Perform a search
    while True:
        query = input("\nEnter your search query (or type 'exit' to quit): ")
        if query.lower() == 'exit':
            break
        
        # Preprocess query and search
        processed_query = preprocess_text(query)
        results = search(processed_query, tfidf_matrix, vectorizer)
        
        # Display search results
        if results:
            print("\nSearch Results:")
            for doc_id, score in results[:5]:  # Show top 5 results
                print(f"Document {doc_id} - Score: {score}")
                print(f"Preview: {documents[doc_id][:200]}...\n")
        else:
            print("No matching documents found.")
    
    print("Exiting search engine. Goodbye!")

if __name__ == "__main__":
    main()
