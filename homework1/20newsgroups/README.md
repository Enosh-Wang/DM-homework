# 20newsgroups
参照Ｋ-means的示例进行实验
## K-means
```
n_samples: 3387, n_features: 10000

Homogeneity: 0.503
Completeness: 0.531
V-measure: 0.516
Adjusted Rand-Index: 0.440
Silhouette Coefficient: 0.007

Top terms per cluster:
Cluster 0: henry space toronto nasa shuttle gov zoo hst spencer mission
Cluster 1: god com sandvik people jesus keith don morality say sgi
Cluster 2: com space access article just posting university digex like host
Cluster 3: graphics image thanks university file files 3d gif program help
```
## AffinityPropagation
```
Clustering sparse data with AffinityPropagation(affinity='euclidean', convergence_iter=15, copy=True,
          damping=0.9, max_iter=200, preference=None, verbose=False)
done in 27.017s

Homogeneity: 0.885
Completeness: 0.191
V-measure: 0.314
Adjusted Rand-Index: 0.008
Silhouette Coefficient: 0.075
```
## DBSCAN
```
Clustering sparse data with DBSCAN(algorithm='auto', eps=0.3, leaf_size=30, metric='euclidean',
    metric_params=None, min_samples=1, n_jobs=None, p=None)
done in 0.483s

Homogeneity: 1.000
Completeness: 0.169
V-measure: 0.289
Adjusted Rand-Index: 0.000
Silhouette Coefficient: 0.005
```
##　MeanShift
因为比较慢，所以调整比较少，效果不太好
```
Clustering sparse data with MeanShift(bandwidth=0.1, bin_seeding=True, cluster_all=True, min_bin_freq=1,
     n_jobs=None, seeds=None)
done in 16.637s

Homogeneity: 0.120
Completeness: 0.057
V-measure: 0.077
Adjusted Rand-Index: 0.043
Silhouette Coefficient: -0.002

Top terms per cluster:
Cluster 0: 107 97 76 northern bell 47 atheism research nntp host
Cluster 1: abo compiled ms fi library pc fortran compile sources ibm
Cluster 2: ithaca eric ia oakland software 510 pp com reached phone
Cluster 3: virginia dobson rwd4f legalize poe acc rob freedom posts unfortunately
```
## SpectralClustering
```
Clustering sparse data with SpectralClustering(affinity='nearest_neighbors', assign_labels='kmeans',
          coef0=1, degree=3, eigen_solver='arpack', eigen_tol=0.0,
          gamma=1.0, kernel_params=None, n_clusters=4, n_init=10,
          n_jobs=None, n_neighbors=10, random_state=None)
done in 2.555s

Homogeneity: 0.655
Completeness: 0.677
V-measure: 0.666
Adjusted Rand-Index: 0.702
Silhouette Coefficient: 0.007
```
## AgglomerativeClustering
```
Clustering sparse data with AgglomerativeClustering(affinity='euclidean', compute_full_tree='auto',
            connectivity=None, linkage='ward', memory=None, n_clusters=4,
            pooling_func='deprecated')
done in 34.188s

Homogeneity: 0.517
Completeness: 0.601
V-measure: 0.556
Adjusted Rand-Index: 0.548
Silhouette Coefficient: 0.006
```
## GaussianMixture
一直内存溢出，跑不出结果
