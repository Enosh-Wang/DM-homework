# digits数据集

## K-means
示例中实验了三种初始化方式，并应用各种指标进行评价。从表格中可以看到PCA降维后能在保证结果质量的情况下大大提高计算速度，取得较好的结果。
```
n_digits: 10,    n_samples 1797,         n_features 64
__________________________________________________________________________________
init            time    inertia homo    compl   v-meas  ARI     AMI     silhouette
k-means++       0.28s   69432   0.602   0.650   0.625   0.465   0.621   0.146
random          0.19s   69694   0.669   0.710   0.689   0.553   0.686   0.147
PCA-based       0.03s   70804   0.671   0.698   0.684   0.561   0.681   0.118
__________________________________________________________________________________
```

## AffinityPropagation

```
Estimated number of clusters: 39
Homogeneity: 0.816
Completeness: 0.561
V-measure: 0.665
Adjusted Rand Index: 0.408
Adjusted Mutual Information: 0.653
Silhouette Coefficient: 0.180
```

## MeanShift

`number of estimated clusters : 10`
# SpectralClustering
```
Estimated number of clusters: 10
Homogeneity: 0.159
Completeness: 0.541
V-measure: 0.245
Adjusted Rand Index: 0.039
Adjusted Mutual Information: 0.234
Silhouette Coefficient: 0.093
```
# Agglomerative Clustering
```
ward :  0.35s
average :       0.25s
complete :      0.24s
single :        0.12s
```

## DBSCAN
```
Estimated number of clusters: 10
Estimated number of noise points: 537
Homogeneity: 0.249
Completeness: 0.415
V-measure: 0.312
Adjusted Rand Index: 0.129
Adjusted Mutual Information: 0.301
Silhouette Coefficient: -0.201
```

# GaussianMixture
```
Estimated number of clusters: 10
Estimated number of noise points: 0
Homogeneity: 0.462
Completeness: 0.475
V-measure: 0.468
Adjusted Rand Index: 0.328
Adjusted Mutual Information: 0.463
Silhouette Coefficient: 0.357
```