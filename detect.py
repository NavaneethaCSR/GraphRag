import nlp
import openai
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler  # Fixed incorrect import (was 'discriminant_analysis')
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import accuracy_score
import pandas as pd
import os
import yaml
from dotenv import load_dotenv
load_dotenv()
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import spacy

nlp = spacy.load("en_core_web_sm")

# Load your settings from a YAML configuration file
with open(r'C:\Users\LENOVO\GRAPHRAG\ragtest1\settings.yaml', 'r') as file:
    settings = yaml.safe_load(file)

# Set up OpenAI API key and models from the configuration
openai.api_key = os.getenv('OPENAI_API_KEY', settings['llm']['api_key'])
embedding_model = settings['embeddings']['llm']['model']
chat_model = settings['llm']['model']

# Function to generate embeddings
def embed_text(text):
    response = openai.embeddings.create(
        model=embedding_model,
        input=text
    )
    return np.array(response.data[0].embedding)

# Function to analyze a single claim for misinformation
def analyze_misinformation(claim, relevant_summary):
    messages = [
        {"role": "user", "content": f"{claim}\n\nSummary: '{relevant_summary}'\nDoes this contain misinformation? Yes or No."}
    ]
    
    response = openai.chat.completions.create(
        model=chat_model,
        messages=messages,
        max_tokens=settings['llm']['max_tokens']
    )
    answer = response.choices[0].message.content
    classification = "misinformation" if "misinformation" in answer.lower() else "information"
    return {
        "classification": classification,
        "evidence": answer
    }

# Function to process input text and extract claims
def process_input(input_text):
    # Load entity specification from your claim extraction guidelines
    entity_specifications = ["organization", "person", "location"]  # example entity types
    
    doc = nlp(input_text)
    claims = []
    
    for ent in doc.ents:
        if ent.label_.lower() in entity_specifications:
            # Assuming each sentence is a potential claim (this can be adjusted for complex patterns)
            for sentence in doc.sents:
                if ent.text in sentence.text:
                    # Example claim structure based on your claim extraction file
                    claims.append({
                        "subject": ent.text.upper(),
                        "object": "NONE",  # This could be populated with additional parsing logic
                        "claim_type": "GENERAL CLAIM",  # Placeholder for claim type logic
                        "claim_status": "UNKNOWN",
                        "claim_description": sentence.text,
                        "claim_date": "NONE",
                        "source_text": sentence.text
                    })
    return claims



def cluster_communities(embeddings, ranks, num_clusters=5):
    # Combine embeddings and ranks into a single feature set
    features = np.hstack((embeddings, ranks.reshape(-1, 1)))  # Stack embeddings with ranks

    # Standardize the features
    scaler = StandardScaler()
    standardized_features = scaler.fit_transform(features)

    # Apply PCA only if there are more than 1 sample and more than 1 feature
    if standardized_features.shape[0] > 1 and standardized_features.shape[1] > 1:
        pca = PCA(n_components=2)
        reduced_features = pca.fit_transform(standardized_features)
    else:
        reduced_features = standardized_features  # Skip PCA if insufficient samples or features

    # Ensure the number of clusters does not exceed the number of samples
    num_clusters = min(num_clusters, standardized_features.shape[0])

    # Perform clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    cluster_labels = kmeans.fit_predict(standardized_features)

    return cluster_labels, reduced_features, kmeans.cluster_centers_  


# Function to find the most relevant community summary for each claim
def find_relevant_community(claim, community_summaries, community_ranks):
    # Embed the claim
    claim_embedding = embed_text(claim)
    similarities = {}

    # Prepare embeddings and ranks for clustering
    embeddings = []
    ranks = []

    for community_id, summary in community_summaries.items():
        summary_embedding = embed_text(summary)
        embeddings.append(summary_embedding)
        ranks.append(community_ranks.get(community_id, 0))  # Get rank from community_ranks

    # Convert embeddings and ranks to numpy arrays
    embeddings = np.array(embeddings)
    ranks = np.array(ranks)

    # Cluster communities based on summaries and ranks
    cluster_labels, reduced_features, cluster_centroids = cluster_communities(embeddings, ranks, num_clusters=5)

    # Extract only the embedding part of centroids for cosine similarity
    embedding_dim = claim_embedding.shape[0]  # Should be 1536
    cluster_centroids_embedding = cluster_centroids[:, :embedding_dim]

    # Find the best cluster for the claim using cosine similarity
    cluster_similarities = [
        cosine_similarity([claim_embedding], [centroid])[0][0] for centroid in cluster_centroids_embedding
    ]
    best_cluster = np.argmax(cluster_similarities)  # Identify the closest cluster

    # Find the most relevant community within the best cluster
    for community_id, summary, label in zip(community_summaries.keys(), community_summaries.values(), cluster_labels):
        if label == best_cluster:
            summary_embedding = embed_text(summary)
            similarities[community_id] = cosine_similarity([claim_embedding], [summary_embedding])[0][0]

    # Identify the best match within the cluster
    best_match = max(similarities, key=similarities.get)
    return best_match, community_summaries[best_match]


# Perform misinformation detection on each claim and classify the entire input
def detect_misinformation(input_text, community_summaries, community_ranks):
    claims = process_input(input_text)
    misinfo_flags = 0
    total_claims = len(claims)
    
    for claim in claims:
        best_community_id, relevant_summary = find_relevant_community(claim['claim_description'], community_summaries, community_ranks)
        result = analyze_misinformation(claim['claim_description'], relevant_summary)
        print(result)
        if result["classification"] == "misinformation":
            misinfo_flags += 1
    
    overall_classification = "Misinformation" if misinfo_flags > total_claims / 2 else "Information"
    return overall_classification

# Load community summaries from CSV into a dictionary
def load_community_summaries(file_path):
    df = pd.read_csv(file_path)
    return {row['community']: row['summary'] for _, row in df.iterrows()}

def load_community_ranks(file_path):
    df = pd.read_csv(file_path)
    return {row['community']: row['rank'] for _, row in df.iterrows()}

# Function to process inputs from a CSV file and save results to another CSV file
def process_inputs_from_csv(input_csv, community_summaries, community_ranks, output_csv):
    print("Processing input CSV...")
    df = pd.read_csv(input_csv)
    predictions = []
    
    for index, row in df.iterrows():
        input_text = row['input_text']
        true_label = row['true_label']
        
        predicted_label = detect_misinformation(input_text, community_summaries, community_ranks)
        print(predicted_label)
        predictions.append({
            "input_text": input_text,
            "predicted_label": predicted_label,
            "true_label": true_label
        })
    
    # Save predictions to output CSV
    output_df = pd.DataFrame(predictions)
    output_df.to_csv(output_csv, index=False)

    # Calculate and print accuracy
    accuracy = accuracy_score(output_df["true_label"], output_df["predicted_label"])
    print(f"Overall Accuracy: {accuracy * 100:.2f}%")

# Example Usage
if __name__ == "__main__":
    print("Starting script...")
    # File paths
    input_csv = r'C:\Users\LENOVO\GRAPHRAG\ragtest1\input_test_100 - .csv'  # Path to the input CSV file
    community_summaries_file = r'C:\Users\LENOVO\GRAPHRAG\ragtest1\output\create_final_community_reports.csv'  # Path to the summaries CSV
 
    output_csv = r'C:\Users\LENOVO\GRAPHRAG\ragtest1\prediction_20.csv'  # Path to the output CSV file

    # Load community summaries and ranks
    community_summaries = load_community_summaries(community_summaries_file)
    community_ranks = load_community_ranks( community_summaries_file)
    
    # Process inputs and save results
    process_inputs_from_csv(input_csv, community_summaries, community_ranks, output_csv)
