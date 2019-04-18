import numpy as np
from scipy.stats import norm
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering

#Negative Binomial
NB_MEAN = 100
NB_VARIANCE = 10000
#Number of sample
M = 100
#
N_COMPONENT = 2
PREVELANCE = 0.05
P1 = 0.1 #Normal
P2 = 0.3 #Abnormal
MEAN_norm = 0
MEAN_abnorm = 10
SD_norm = 1
SD_abnorm = 1

def cal_NB_param(mean,variance):
    p = mean/variance
    r = p*mean/(1-p)
    return r,p


class GMMSequences(object):
    def __init__(self, n=2, p=np.array([0.9, 0.1]), mean=np.array([0, 10]), sd=np.array([1, 1])):
        self.n = n
        self.p = p
        self.mean = mean
        self.sd = sd

    def sample(self, n):
        a = np.random.choice(self.n, n, True, self.p)
        b = np.random.normal(size=n)
        return b * self.sd[a] + self.mean[a]

    def score(self, X):
        return np.sum([self.score_sample(x) for x in X])

    def score_sample(self, x):
        return np.max(np.log(norm.pdf(x, self.mean, self.sd)) + np.log(self.p))


class MixtureGMM(object):
    def __init__(self, k=2, n=2, p_init=np.array([1-PREVELANCE, PREVELANCE]), p=np.array([[0.95, 0.05], [0.8, 0.2]]),
                 mean=np.array([[0, 10], [0, 10]]),
                 sd=np.array([[1, 1], [1, 1]])):
        self.k = k
        self.gmm = [GMMSequences(n, p[i, :], mean[i, :], sd[i, :]) for i in range(k)]
        self.p_init = p_init

    def sample(self, n):
        y = np.random.choice(self.k, 1, p=self.p_init)[0]
        return self.gmm[y].sample(n)[:, None], y

    def generate(self, m, mean, variance):
        data = []
        y = []
        r, p = cal_NB_param(mean, variance)
        for _ in range(m):
            N = 0
            while N < 30:
                N = np.random.negative_binomial(r, p, 1)[0]
            temp = self.sample(N)
            data.append(temp[0])
            y.append(temp[1])
        return data, np.array(y)


class GmmCluster(object):
    def __init__(self):
        self.models = []
        self.X = None
        self.likelihood_matrix = None
        self.distance_matrix = None
        self.clustering = None

    def fit(self, X, n_component=N_COMPONENT, n_clusters=2):
        self.X = X
        self.models = [GaussianMixture(n_component).fit(X[i]) for i in range(len(X))]
        self.likelihood_matrix = np.zeros((len(X), len(X)))
        self.distance_matrix = np.zeros((len(X), len(X)))

        for i in range(len(X)):
            for j in range(len(X)):
                self.likelihood_matrix[i, j] = self.models[j].score(self.X[i])
        for i in range(len(X)):
            for j in range(i + 1, len(X)):
                self.distance_matrix[i, j] = self.distance_matrix[j, i] = \
                    -(self.likelihood_matrix[i, j]+self.likelihood_matrix[j, i])/2
        self.clustering = AgglomerativeClustering(n_clusters=n_clusters, affinity="precomputed", linkage="average").fit(
            self.distance_matrix)
        return self.clustering.labels_


def data_trans(data, cluster, n_cluster=2):
    res = [np.array([])[:, None] for _ in range(n_cluster)]
    for i in range(len(data)):
        res[cluster[i]] = np.vstack([res[cluster[i]], data[i]])
    return res


def kmeans(data, n_cluster=2, n_components=2, n_iter=100):
    n = len(data)
    cluster = np.random.choice(n_cluster, n)
    while len(set(cluster)) != n_cluster:
        cluster = np.random.choice(2, n)
    res = [GaussianMixture(n_components) for _ in range(n_cluster)]
    for _ in range(n_iter):
        #M
        temp = data_trans(data, cluster, n_cluster)
        for k in range(n_cluster):
            res[k].fit(temp[k])
        #E
        cluster = np.argmax(np.array([[res[k].score(data[i]) for k in range(n_cluster)] for i in range(len(data))]), axis=1)
    return res, cluster


def evaluate(y, cluster,penalty=10):
    return min(penalty*np.sum((y==1)*(cluster==0))+np.sum((y==0)*(cluster==1)),
               penalty * np.sum((y == 1) * (cluster == 1)) + np.sum((y == 0) * (cluster == 0)))