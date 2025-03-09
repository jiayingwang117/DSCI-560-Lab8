# DSCI-560-Lab8

This project implements **document clustering** using **Doc2Vec** and **Word2Vec**, allowing for comparison between **document-level** and **word-level embeddings**.

---

### Dependencies
Ensure you have the following Python libraries installed:

```bash
pip3 install numpy pandas matplotlib gensim scikit-learn joblib scikit-learn-extra nltk
```

---

### Run the script:

**To run Doc2Vec**
```bash
python3 doc2vec.py
```

**To run Word2Vec**
```bash
python3 word2vec.py
```

The script will:
   - Train **Doc2Vec/Word2Vec** models with different vector sizes.
   - Generate document embeddings.
   - Perform clustering on document embeddings.
   - Extract keywords per cluster.
   - Generate visualizations saved in `./cluster_visualizations/`.
   - Save trained models in the `models/` directory.

The **best configuration** will be displayed based on the silhouette score.

---

### 📂 Output Files
📊 **Cluster visualizations** (`.png`) stored in `./cluster_visualizations/`  
💾 **Trained models** (`.pkl`) stored in `./models/`  
🏆 **Best-performing model** determined by **silhouette score**  

---
### Doc2Vec Implementation (doc2vec.py)

#### Configurations
We test three different **Doc2Vec** configurations:

**Config_VS_50** → Vector size = 50  
**Config_VS_100** → Vector size = 100  
**Config_VS_150** → Vector size = 150  

#### Pipeline
1.	Load Reddit posts from a CSV file.
2.	Combine title, selftext, and OCR-extracted text.
3.	Train Doc2Vec to generate document embeddings.
4.	Cluster documents using KMedoids.
5.	Extract keywords per cluster using TF-IDF.
6.	Visualize the clusters using PCA.

### Word2Vec Implementation (doc2vec.py)

#### Configurations
We test three different **Word2Vec** configurations:

**Config_VS_50** → Vector size = 50  
**Config_VS_100** → Vector size = 100  
**Config_VS_150** → Vector size = 150  


#### Pipeline
1.	Load Reddit posts from a CSV file.
2.	Preprocess text by tokenization & stopword removal.
3.	Train Word2Vec model on individual words.
4.	Convert posts into document vectors by averaging word embeddings.
5.	Cluster documents using KMedoids with cosine similarity.
6.	Extract keywords per cluster using TF-IDF.
7.	Visualize the clusters using PCA.

**🔧 Adjusting the number of clusters (We use 3 here)**: Change the `num_clusters` parameter in the script.
---