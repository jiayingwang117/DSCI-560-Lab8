# DSCI-560-Lab8


## doc2vec.py

---

### Dependencies
Ensure you have the following Python libraries installed:

```bash
pip3 install numpy pandas matplotlib gensim scikit-learn joblib scikit-learn-extra
```

---

### Run the script:

```bash
python3 doc2vec.py
```

The script will:
   - Train **Doc2Vec** models with different vector sizes.
   - Perform clustering on document embeddings.
   - Identify key messages and keywords per cluster.
   - Generate visualizations saved in `./cluster_visualizations/`.
   - Save trained models in the `models/` directory.

The **best configuration** will be displayed based on the silhouette score.

---

### Configurations
We test three different **Doc2Vec** configurations:

**Config_VS_50** → Vector size = 50  
**Config_VS_100** → Vector size = 100  
**Config_VS_150** → Vector size = 150  

**🔧 Adjusting the number of clusters (We use 3 here)**: Change the `num_clusters` parameter in the script.

---

### 📂 Output Files
📊 **Cluster visualizations** (`.png`) stored in `./cluster_visualizations/`  
💾 **Trained models** (`.pkl`) stored in `./models/`  
🏆 **Best-performing model** determined by **silhouette score**  

---

