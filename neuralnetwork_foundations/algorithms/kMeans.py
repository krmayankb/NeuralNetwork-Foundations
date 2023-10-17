import random
import numpy as np
random.seed(42)

class kMeans():
    def __init__(self, num_clusters=3, max_iter=100, tol=0.0001): 
        self.num_clusters = num_clusters
        self.max_iter = max_iter
        self.tol = tol

    def fit(self, data):
        # initialize centroids
        self.centroids = np.array([random.choice(data) for _ in range(self.num_clusters)])
        
        for iteration in range(self.max_iter): 
            self.iteration = iteration
            # initialize clusters
            self.clusters = {i:[] for i in range(self.num_clusters)}
            for point in data: 
                # assign each point to the closest cluster
                distances = self.distance(point, self.centroids)
                closest_cluster = np.argmin(distances)
                self.clusters[closest_cluster].append(point)
            
            # update centroids
            prev_centroids = self.centroids
            self.centroids = np.array([np.mean(self.clusters[i], axis=0) for i in range(self.num_clusters)])
    
            # check if converged
            total_distance = 0
            for center in zip(prev_centroids, self.centroids):
                total_distance += np.sqrt(np.sum((center[0]-center[1])**2))
            
            if total_distance <= self.tol:
                break

    def distance(self, point, data, metric="euclidean"):
        if metric == "euclidean":            
            return np.sqrt(np.sum((point-data)**2, axis=1))
        else:
            raise NotImplementedError
    
    def predict(self, data):
        clusters = []
        for point in data: 
            distance = self.distance(point, self.centroids)
            clusters.append(np.argmin(distance))
        return clusters
    
if __name__ == "__main__":
    a = np.random.rand(100, 2)
    kmeans = kMeans(num_clusters=5, max_iter=100)
    kmeans.fit(a)
