import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids 

# Step 1: Load posts from CSV file
def load_posts(csv_file):
    """
    Load posts from a CSV file.
    The CSV should contain columns: 'id', 'clean_title', 'clean_selftext', and 'ocr_text'.
    Returns a list of post dictionaries and the original DataFrame.
    """
    df = pd.read_csv(csv_file)
    posts = df.to_dict(orient='records')
    return posts, df

# Step 2: Prepare document texts (combine title, selftext, and OCR text)
def prepare_documents(posts):
    """
    Combine 'clean_title', 'clean_selftext', and 'ocr_text' (if available)
    into one string per post.
    """
    docs = []
    for post in posts:
        text = f"{post['clean_title']} {post['clean_selftext']}"
        if 'ocr_text' in post and post['ocr_text']:
            ocr_text_value = post['ocr_text']
            if isinstance(ocr_text_value, str): 
                if ocr_text_value.strip().lower() != "none":
                    text += " " + ocr_text_value
            elif not pd.isna(ocr_text_value): 
                text += " " + str(ocr_text_value) # Convert non-string, non-NaN value to string
        docs.append(text)
    return docs

# Step 3: Train a Doc2Vec model on the documents
def compute_doc2vec_embeddings(docs, vector_size=100, min_count=1, epochs=40):
    """
    Train a Doc2Vec model on the provided documents and return the embedding
    for each document along with the trained model.
    """
    tagged_docs = [TaggedDocument(doc.split(), [i]) for i, doc in enumerate(docs)]
    model = Doc2Vec(vector_size=vector_size, window=5, min_count=min_count, workers=4, epochs=epochs)
    model.build_vocab(tagged_docs)
    model.train(tagged_docs, total_examples=model.corpus_count, epochs=model.epochs)
    embeddings = np.array([model.dv[i] for i in range(len(docs))])
    return embeddings, model

# Step 4: Cluster the embeddings using KMedoids with cosine distance
def cluster_embeddings(embeddings, num_clusters=5):
    """
    Cluster the embeddings using KMedoids with cosine distance.
    Returns cluster labels and the KMedoids model.
    """
    kmedoids = KMedoids(n_clusters=num_clusters, metric='cosine', random_state=42) # Using KMedoids with cosine metric
    labels = kmedoids.fit_predict(embeddings) 
    return labels, kmedoids

# Step 5a: Find the closest message to each cluster centroid
def find_closest_messages(embeddings, labels, medoids_indices): 
    """
    For each cluster, find the document whose embedding is the medoid.
    Returns a dictionary mapping cluster index to the document index of the medoid.
    """
    closest = {}
    # Medoids are actual data points, so we use their indices directly.
    for cluster in np.unique(labels):
        closest_index = medoids_indices[cluster] 
        closest[cluster] = closest_index
    return closest

# Step 5b: Extract keywords from a cluster using TF-IDF
def extract_keywords_for_cluster(texts, top_n=5):
    """
    Extract keywords from a list of texts using TF-IDF.
    Returns a list of the top_n keywords.
    """
    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)
    scores = np.array(tfidf_matrix.mean(axis=0)).flatten()
    indices = scores.argsort()[-top_n:][::-1]
    feature_names = np.array(vectorizer.get_feature_names_out())
    keywords = feature_names[indices]
    return list(keywords)

def get_cluster_keywords(docs, labels, num_clusters=5):
    """
    For each cluster, extract keywords from all documents within that cluster.
    Returns a dictionary mapping cluster index to its list of keywords.
    """
    cluster_keywords = {}
    for cluster in range(num_clusters):
        cluster_docs = [docs[i] for i, label in enumerate(labels) if label == cluster]
        if cluster_docs:
            keywords = extract_keywords_for_cluster(cluster_docs, top_n=5)
        else:
            keywords = []
        cluster_keywords[cluster] = keywords
    return cluster_keywords

# Step 6: Visualize clusters in 2D using PCA
def visualize_clusters(embeddings, labels, cluster_keywords, closest_messages, config_name):
    """
    Reduce embeddings to 2D using PCA and visualize clusters with KMedoids medoids.
    The medoid message in each cluster is marked and clusters are annotated with keywords.
    """
    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    # Create output directory if it doesn't exist
    output_dir = './cluster_visualizations'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c=labels,
                            cmap='viridis', alpha=0.6, edgecolor='k')
    plt.colorbar(scatter, label="Cluster Label")

    unique_labels = np.unique(labels)
    for cluster in unique_labels:
        indices = np.where(labels == cluster)[0]
        cluster_points = embeddings_2d[indices]
        
        # In KMedoids, medoids are actual data points, not centroids.
        medoid_index_in_cluster = closest_messages[cluster] 
        medoid_point = embeddings_2d[medoid_index_in_cluster] 

        # Annotate medoid and cluster keywords
        keywords_text = ", ".join(cluster_keywords[cluster])
        plt.text(medoid_point[0], medoid_point[1], f"Cluster {cluster}\n{keywords_text}", 
                 fontsize=9, ha='center', va='center',
                 bbox=dict(facecolor='white', alpha=0.7, edgecolor='black', boxstyle='round'))
        closest_index = closest_messages[cluster]
        plt.scatter(embeddings_2d[closest_index, 0], embeddings_2d[closest_index, 1],
                    c='red', marker='x', s=100, linewidths=2)

    # Save the plot
    plt.title(f"Message Clusters Visualization (PCA) - {config_name} (KMedoids)") 
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/cluster_visualization_{config_name}_kmedoids.png", dpi=300, bbox_inches='tight') 
    plt.close()

# Main function to run the pipeline
def main():
    posts, df = load_posts("./reddit_data/cleaned_posts.csv")
    if not posts:
        print("No posts found in the CSV file.")
        return

    # Prepare document texts
    docs = prepare_documents(posts)

    # Define three Doc2Vec configurations
    configs = [
        {"name": "Config_VS_50", "vector_size": 50, "min_count": 1, "epochs": 40},
        {"name": "Config_VS_100", "vector_size": 100, "min_count": 1, "epochs": 40},
        {"name": "Config_VS_150", "vector_size": 150, "min_count": 1, "epochs": 40}
    ]

    # Number of clusters for KMedoids(can be changed)
    num_clusters = 2  
    results = []  

    # Iterate over configurations
    for cfg in configs:
        print(f"\n--- {cfg['name']} ---")
        
        # Compute embeddings using current configuration
        embeddings, d2v_model = compute_doc2vec_embeddings(
            docs,
            vector_size=cfg["vector_size"],
            min_count=cfg["min_count"],
            epochs=cfg["epochs"]
        )
        
        # Normalize embeddings for cosine similarity clustering
        norm_embeddings = normalize(embeddings)
        
        # Cluster the normalized embeddings using KMedoids
        labels, kmedoids_model = cluster_embeddings(norm_embeddings, num_clusters=num_clusters) 
        medoids_indices = kmedoids_model.medoid_indices_
        
        # Compute silhouette score with cosine metric 
        silhouette = silhouette_score(norm_embeddings, labels, metric="cosine")
        print(f"Silhouette Score (Cosine): {silhouette:.3f}")
        
        # Identify the closest messages to the medoids
        closest_messages = find_closest_messages(norm_embeddings, labels, medoids_indices)
        
        # Extract keywords for each cluster
        cluster_keywords = get_cluster_keywords(docs, labels, num_clusters)
        
        # Visualize the clusters
        visualize_clusters(norm_embeddings, labels, cluster_keywords, closest_messages, cfg["name"])
        results.append({"config": cfg["name"], "silhouette": silhouette, "model": kmedoids_model, "d2v_model": d2v_model})
        
        # Save models
        if not os.path.exists("models"):
            os.makedirs("models")

        joblib.dump(kmedoids_model, f"models/{cfg['name']}_kmedoids_model.pkl")
        joblib.dump(d2v_model, f"models/{cfg['name']}_d2v_model.pkl")


    # Find the best configuration based on silhouette score
    best_config = max(results, key=lambda x: x['silhouette'])
    print(f"\nBest Configuration: {best_config['config']} with Silhouette Score: {best_config['silhouette']:.3f}")

if __name__ == "__main__":
    main()
