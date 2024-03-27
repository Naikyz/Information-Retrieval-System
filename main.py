import argparse
import re
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import json
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import unicodedata

data_path = "."

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

# Document collection and reading
def readfile(path, filename):
    '''
    Reads file contents
    Parameters:
    path (str) : File path
    filename : File to be read
    Returns:
        file_as_string (str) : file contents as string
    '''
    with open(path + filename) as file:
        file_as_list = file.readlines()
        file_as_string = ''.join(map(str, file_as_list))

    return file_as_string

# Pre-processing of text and generation of normalized tokens
def preprocess(text):
    '''
    Convert text file into list of normalized tokens, handling accents and keeping contractions like "let's" intact.
    Parameters:
        text (str) : text to be preprocessed
    Returns:
        normalized (list) : list of normalized tokens
    '''
    normalized = []
    # Normalize unicode characters to decompose accents from letters
    text = unicodedata.normalize('NFD', text)
    text = text.encode('ascii', 'ignore').decode('utf-8')

    # Remove specific characters but keep apostrophes for contractions
    text = text.replace('"', '').replace(".", "")
    text = text.lower()
    # Remove numbers
    text = re.sub(r'\b[0-9]+\b', '', text)

    tokens = text.split()

    # Assuming English stopwords, for French or other languages use appropriate stopwords list
    stop_words = set(stopwords.words('english'))

    # Filter out stop words and remove punctuation except apostrophes in contractions
    for word in tokens:
        if word not in stop_words:
            # Remove all punctuation except apostrophes
            word = re.sub(r'[^\w\s\'-]', '', word)
            if word:  # Check if word is not empty
                normalized.append(word)
    return normalized



# Create a mapping of terms to documents
import os

def create_index(terms, docID):
    '''
    Creates or updates an Inverted Index and saves it in a json file.
    Parameters:
        terms (list) : List of terms to be indexed
        docID (str) : Unique document ID
    '''
    ivdict = {}  # Initialize an empty dictionary

    # Check if the file exists
    index_path = os.path.join(data_path, 'dict.json')
    if os.path.isfile(index_path):
        with open(index_path, 'r') as index_file:
            ivdict = json.load(index_file)  # Load existing data

    for term in terms:
        if term in ivdict:
            postings_list = [posting[0] for posting in ivdict[term]]
            if docID in postings_list:
                next(posting for posting in ivdict[term] if posting[0] == docID)[1] += 1
            else:
                ivdict[term].append([docID, 1])
        else:
            ivdict[term] = [[docID, 1]]

    with open(index_path, 'w') as index_file:
        json.dump(ivdict, index_file)


### GENERATE INVERTED INDEX DOC

documents_path = "./documents/"

document_files = os.listdir(documents_path)

for doc_file in document_files:
    if not os.path.isfile(os.path.join(documents_path, doc_file)):
        continue

    # Read the content of the file
    text_content = readfile(documents_path, doc_file)

    # Preprocess the content to get tokens
    tokens = preprocess(text_content)

    # Create or update the index using the file name (without extension) as docID
    docID = os.path.splitext(doc_file)[0]
    create_index(tokens, docID)

## BOOLEAN QUERYING

# Load the inverted index
def load_inverted_index(index_path):
    if os.path.exists(index_path):
        with open(index_path, 'r') as f:
            return json.load(f)
    return {}

# Preprocess a term similar to document preprocessing
def preprocess_term(term):
    term = term.lower()  # Lowercase
    term = re.sub(r'[^\w\s\']', '', term)  # Remove punctuation, keep apostrophes
    # Tokenization and stopword removal are not applied here as we're processing term-by-term
    return term

# Evaluate a simple Boolean query against the inverted index
def evaluate_boolean_query(query, inverted_index):
    terms = word_tokenize(query)
    ops = {'AND', 'OR', 'NOT'}  # Define Boolean operators
    processed_terms = [(preprocess_term(term) if term not in ops else term) for term in terms]

    current_docs = set()
    current_op = None

    for term in processed_terms:
        if term in ops:
            current_op = term
        else:
            # Adjust this line: Extract just the document IDs from each term's postings
            term_docs = set([doc[0] for doc in inverted_index.get(term, [])])
            if current_op is None:
                current_docs = term_docs
            elif current_op == 'AND':
                current_docs = current_docs.intersection(term_docs)
            elif current_op == 'OR':
                current_docs = current_docs.union(term_docs)
            elif current_op == 'NOT':
                current_docs = current_docs.difference(term_docs)

    return current_docs

# Assuming `documents` is a list where each item is the text of a document
def compute_tfidf(documents):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(documents)
    return tfidf_matrix, vectorizer.get_feature_names_out()

def calculate_cosine_similarity(tfidf_matrix, query_vector):
    cosine_similarities = cosine_similarity(tfidf_matrix, query_vector)
    return cosine_similarities

def vector_space_query(query, tfidf_matrix, feature_names):
    # Convert the query into a TF-IDF vector using the same feature set as the documents
    query_vector = TfidfVectorizer(vocabulary=feature_names).fit_transform([query])
    cosine_similarities = calculate_cosine_similarity(tfidf_matrix, query_vector)
    return np.argsort(cosine_similarities.flatten())[::-1]  # Sort documents by similarity


# Example Usage:
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Information Retrieval System')
    parser.add_argument('--boolean', type=str, help='Boolean query to be executed')
    parser.add_argument('--vector', type=str, help='Vector space model query')

    args = parser.parse_args()

    import os

# Assuming the rest of your script is as previously discussed...

    if args.vector:
        query = args.vector

        # Initialize lists to hold the preprocessed documents and their names
        documents = []
        document_names = []  # List to store the names of the documents

        # Ensure the documents_path points to your directory containing the documents
        documents_path = "./documents/"

        # List all document files in the directory
        document_files = os.listdir(documents_path)

        # Read and preprocess each document
        for doc_file in document_files:
            full_path = os.path.join(documents_path, doc_file)
            if not os.path.isfile(full_path):
                continue  # Skip if not a file

            # Load the document content
            text_content = readfile(documents_path, doc_file)

            # Preprocess the text content and concatenate it into a single string
            processed_text = ' '.join(preprocess(text_content))

            # Append the preprocessed text and the document's name to their respective lists
            documents.append(processed_text)
            document_names.append(doc_file)  # Store the document name

        # Compute TF-IDF matrix for the loaded documents
        tfidf_matrix, feature_names = compute_tfidf(documents)

        # Perform the vector space query
        sorted_doc_indices = vector_space_query(query, tfidf_matrix, feature_names)

        # Map the sorted document indices back to their names using the document_names list
        sorted_doc_names = [document_names[index] for index in sorted_doc_indices]

        print("Documents sorted by relevance:", sorted_doc_names)

    elif args.boolean:
        query = args.boolean
        index_path = './dict.json'
        inverted_index = load_inverted_index(index_path)
        result_docs = evaluate_boolean_query(query, inverted_index)
        print("Documents matching query:", result_docs)
    else:
        print("No query provided. Please use --boolean 'QUERY' to perform a search.")