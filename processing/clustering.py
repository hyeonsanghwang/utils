from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering


def clustering_kmeans(x, n=2):
    kmeans = KMeans(n_clusters=n)
    res = kmeans.fit_predict(x)
    return res


def clustering_dbscan(x, eps=0.1, min_samples=10):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    res = dbscan.fit_predict(x)
    return res


def clustering_gaussian_mixture(x, n=2):
    gm = GaussianMixture(n_components=n)
    res = gm.fit_predict(x)
    return res


def clustering_hierarchy(x, n=2, affinity='euclidean', linkage='complete'):
    """
    affinity : euclidean, l1, l2, manhattan, cosine, precomputed (linkage = ward -> euclidean)
    linkage : ward, complete, average, single
    """
    ac = AgglomerativeClustering(n_clusters=n, affinity=affinity, linkage=linkage)
    res = ac.fit_predict(x)
    return res
