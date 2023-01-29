# sgonzalezsilotPortfolio
Data Science Portfolio

## [Project 1: Pneumonia Detection with Chest X-Ray Images](https://github.com/sgonzalezsilot/FinalProjectComputerVision): 
* Build a CNN (Convolutional Neural Network) to detect pneumonia with chest x-ray images.
* Fine-Tuned Resnet50 (trained with Imagenet) to obtain 94.54% accuracy.
* We apply ImageDataGenerator to balance the classes.
* We used TensorFlow and Scikit-Learn.

![](images/ROC.png)

![](images/matriz.png)


## [Project 2: Analysis of +4000 TED Talks topics using document clustering](https://github.com/sgonzalezsilot/TedTalksClustering): 
* Comparison of clustering formed using tf-idf and word embeddings using the most commons clustering algorithms like KMeans, Gaussian Mixture Models and Agglomerative Clustering.
* Tuning of the hyperparameters of all models.
* Comparaison of the results using multiple clustering metrics (DBI, Silhoutte and Calinski).
* Bonus experiment using only the most relevant tf-idf words and partly solving the curse of dimensionality.
* Bonus experiment using word embeddings from Microsoft MiniLM-L12-H384.
* Final analysis using wordclouds and n-grams to identify the topics.
* Found insights about which algorithms and metrics work best for document clustering and why.
* I used cuML, Spark (PySpark) and sentence-transformers.

![](images/Clusters_KMeans.png)

![](images/Clusters_GMM.png)
