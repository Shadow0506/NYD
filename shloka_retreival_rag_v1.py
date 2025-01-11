# Import necessary libraries
import os
import json
import pandas as pd
import chromadb
from chromadb.utils import embedding_functions
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Set up ChromaDB client
client = chromadb.Client(Settings(chroma_db_impl="duckdb+parquet", persist_directory="./chromadb"))

# Prepare the embedding function
embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def embed_text(texts):
    return embedding_model.encode(texts)

embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# Load the datasets

# Assuming datasets are in CSV format and located in 'data' folder
gita_file = 'data/Bhagwad_Gita_Verses_English.csv'
pys_file = 'data/Patanjali_Yoga_Sutras_Verses_English.csv'

# Load data into pandas DataFrames
gita_df = pd.read_csv(gita_file)
pys_df = pd.read_csv(pys_file)

# Combine datasets
combined_df = pd.concat([gita_df, pys_df], ignore_index=True)

# Ensure necessary columns are present
# For example, 'id', 'text', 'chapter', 'verse', 'book'

# Preprocessing - cleaning text data if necessary
def preprocess_text(text):
    # Implement any necessary text preprocessing here
    # For example, removing extra whitespace, special characters etc.
    return text.strip()

combined_df['text'] = combined_df['text'].apply(preprocess_text)

# Create or get a collection in ChromaDB
collection_name = 'shlokas_collection'
collection = client.get_or_create_collection(
    name=collection_name,
    embedding_function=embedding_function
)

# Check if collection is empty, if so, insert data
if len(collection.get()['ids']) == 0:
    # Prepare data for insertion into ChromaDB
    texts = combined_df['text'].tolist()
    ids = combined_df.index.astype(str).tolist()  # Ensure ids are strings
    metadatas = combined_df.to_dict('records')

    # Insert data into ChromaDB
    collection.add(
        documents=texts,
        metadatas=metadatas,
        ids=ids
    )

# Function to check for inappropriate queries
def is_inappropriate(query):
    # Simple keyword-based check; can be replaced with more advanced methods
    inappropriate_keywords = ['violence', 'hate', 'illegal', 'explicit']
    query_lower = query.lower()
    return any(keyword in query_lower for keyword in inappropriate_keywords)

# Function to process user query
def process_query(query):
    # Check for inappropriate content
    if is_inappropriate(query):
        return {
            "error": "Your query contains inappropriate content and cannot be processed."
        }

    # Embed the query
    query_embedding = embedding_function.embed_documents([query])[0]

    # Search for similar shlokas
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5,
        include=['metadatas', 'documents', 'distances']
    )

    # Retrieve top results
    top_results = results['documents'][0]
    top_metadatas = results['metadatas'][0]
    top_distances = results['distances'][0]

    # Prepare context for LLM
    retrieved_shlokas = ''
    for idx, (shloka, metadata) in enumerate(zip(top_results, top_metadatas)):
        source = metadata.get('book', 'Unknown Source')
        chapter = metadata.get('chapter', '')
        verse = metadata.get('verse', '')
        shloka_info = f"Source: {source}, Chapter: {chapter}, Verse: {verse}"
        retrieved_shlokas += f"Shloka {idx+1} ({shloka_info}):\n{shloka}\n\n"

    # Prepare prompt
    prompt = f"""
You are an expert in Vedanta philosophy.

User Query:
{query}

Relevant Shlokas:
{retrieved_shlokas}

Based on the above shlokas, please provide a concise and accurate answer to the user's query. Ensure that your answer is based only on the provided shlokas and avoid any hallucinations or unsupported claims.
"""

    # Load the LLM model and tokenizer
    # For example, using 'gpt2' for simplicity
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2", pad_token_id=tokenizer.eos_token_id)

    # Generate answer
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    outputs = model.generate(
        inputs,
        max_length=512,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=2
    )
    generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Extract the answer
    answer_start = generated_answer.find("Based on the above shlokas") + len("Based on the above shlokas, please provide a concise and accurate answer to the user's query. Ensure that your answer is based only on the provided shlokas and avoid any hallucinations or unsupported claims.")
    answer = generated_answer[answer_start:].strip()

    # Prepare output JSON
    output = {
        "query": query,
        "retrieved_shlokas": retrieved_shlokas,
        "answer": answer
    }

    return output

# Example usage
if __name__ == "__main__":
    user_query = input("Enter your query: ")
    output = process_query(user_query)
    # Output in JSON format
    print(json.dumps(output, indent=2, ensure_ascii=False))