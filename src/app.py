from  extractors.base_extractor import extract_text
from  preprocess.text_preprocessor import preprocess_text
from  index.indexer import create_index
from  search.search_engine import search
import nltk

# Ensure necessary NLTK data is available
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')

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
