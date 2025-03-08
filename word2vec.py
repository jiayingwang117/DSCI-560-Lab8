import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models import Word2Vec
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import re

nltk.download('punkt_tab')
nltk.download('stopwords')

# Step 1: Load posts from CSV file
def load_posts(csv_file):
    df = pd.read_csv(csv_file)
    posts = df.to_dict(orient='records')
    return posts, df

# Step 2: Preprocess text (tokenization, cleaning)
def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', str(text).lower())  # Remove special characters
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords.words('english')]  # Remove stopwords
    return tokens

# Step 3: Prepare document texts (combine title, selftext, and OCR text)
def prepare_documents(posts):
    docs = []
    for post in posts:
        text = f"{post['clean_title']} {post['clean_selftext']}"
        if 'ocr_text' in post and isinstance(post['ocr_text'], str) and post['ocr_text'].strip().lower() != "none":
            text += " " + post['ocr_text']
        docs.append(preprocess_text(text))
    return docs

# Step 4: Train a Word2Vec model
def compute_word2vec_embeddings(docs, vector_size=100, min_count=1, epochs=40):
    model = Word2Vec(sentences=docs, vector_size=vector_size, window=5, min_count=min_count, workers=4, epochs=epochs)
    word_vectors = model.wv  # Get trained word vectors
    return word_vectors, model

# Step 5: Convert posts into document vectors (average word embeddings)
def get_document_vectors(docs, word_vectors, vector_size=100):
    doc_vectors = []
    for tokens in docs:
        valid_tokens = [word_vectors[word] for word in tokens if word in word_vectors]
        if valid_tokens:
            doc_vectors.append(np.mean(valid_tokens, axis=0))
        else:
            doc_vectors.append(np.zeros(vector_size))  # If no valid words, use a zero vector
    return np.array(doc_vectors)

# Step 6: Cluster the embeddings using KMedoids
def cluster_embeddings(embeddings, num_clusters=5):
    kmedoids = KMedoids(n_clusters=num_clusters, metric='cosine', random_state=42)
    labels = kmedoids.fit_predict(embeddings)
    return labels, kmedoids

# Step 7a: Find the closest message to each cluster medoid
def find_closest_messages(embeddings, labels, medoids_indices): 
    closest = {cluster: medoids_indices[cluster] for cluster in np.unique(labels)}
    return closest

# Step 7b: Extract keywords from clusters using TF-IDF
def extract_keywords_for_cluster(texts, top_n=5):
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    indices = scores.argsort()[-top_n:][::-1]
    feature_names = np.array(vectorizer.get_feature_names_out())
    return feature_names[indices].tolist()

def get_cluster_keywords(docs, labels, num_clusters=5):
    cluster_keywords = {}
    for cluster in range(num_clusters):
        cluster_docs = [" ".join(docs[i]) for i, label in enumerate(labels) if label == cluster]
        cluster_keywords[cluster] = extract_keywords_for_cluster(cluster_docs) if cluster_docs else []
    return cluster_keywords

# Step 8: Visualize clusters in 2D using PCA
def visualize_clusters(embeddings, labels, cluster_keywords, closest_messages, config_name):
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    output_dir = './cluster_visualizations'
    os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels, cmap='viridis', alpha=0.6, edgecolor='k')
    plt.colorbar(scatter, label="Cluster Label")

    for cluster in np.unique(labels):
        medoid_index = closest_messages[cluster] 
        medoid_point = embeddings_2d[medoid_index]  
        keywords_text = ", ".join(cluster_keywords[cluster])
        
        plt.text(medoid_point[0], medoid_point[1], f"Cluster {cluster}\n{keywords_text}",
                 fontsize=9, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round'))
        plt.scatter(medoid_point[0], medoid_point[1], c='red', marker='x', s=100, linewidths=2)

    plt.title(f"Message Clusters Visualization (PCA) - {config_name} (KMedoids)") 
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_visualization_{config_name}_kmedoids.png", dpi=300, bbox_inches='tight')
    plt.close()

# Main function to run the pipeline
def main():
    posts, df = load_posts("reddit_data/cleaned_posts.csv")
    if not posts:
        print("No posts found in the CSV file.")
        return

    docs = prepare_documents(posts)

    # Define Word2Vec Configurations
    configs = [
        {"name": "Config_WV_50", "vector_size": 50, "min_count": 1, "epochs": 40},
        {"name": "Config_WV_100", "vector_size": 100, "min_count": 1, "epochs": 40},
        {"name": "Config_WV_150", "vector_size": 150, "min_count": 1, "epochs": 40}
    ]

    num_clusters = 2  
    results = []  

    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")

        # Train Word2Vec model
        word_vectors, w2v_model = compute_word2vec_embeddings(docs, vector_size=cfg["vector_size"], min_count=cfg["min_count"], epochs=cfg["epochs"])

        # Convert posts to document vectors
        doc_vectors = get_document_vectors(docs, word_vectors, vector_size=cfg["vector_size"])
        norm_embeddings = normalize(doc_vectors)

        # Cluster the embeddings
        labels, kmedoids_model = cluster_embeddings(norm_embeddings, num_clusters=num_clusters)
        medoids_indices = kmedoids_model.medoid_indices_

        silhouette = silhouette_score(norm_embeddings, labels, metric="cosine")
        print(f"Silhouette Score (Cosine): {silhouette:.3f}")

        closest_messages = find_closest_messages(norm_embeddings, labels, medoids_indices)
        cluster_keywords = get_cluster_keywords(docs, labels, num_clusters)
        visualize_clusters(norm_embeddings, labels, cluster_keywords, closest_messages, cfg["name"])

        results.append({"config": cfg["name"], "silhouette": silhouette})

    best_config = max(results, key=lambda x: x['silhouette'])
    print(f"\nBest Configuration: {best_config['config']} with Silhouette Score: {best_config['silhouette']:.3f}")

if __name__ == "__main__":
    main()