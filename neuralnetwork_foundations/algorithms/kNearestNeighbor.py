import numpy as np 

class KNearstNeighbors(): 
    def __init__(self, n_neighbors=5, dist_metric="euclidean"):
        self.n_neighbors = n_neighbors
        self.dist_metric = dist_metric

    def fit(self, X_train, y_train):
        self.X = X_train
        self.y = y_train
    
    
    def distance(self, point, data, metric="euclidean"):
        """
        Returns all the distances between a point and the data
        """
        if metric == "euclidean":
            return np.sqrt(np.sum((point-data)**2, axis=1))   
        else: 
            raise ValueError("Invalid distance metric")     
    
    def most_common(self, labels, using="numpy"):
        """
        Returns most common class among the labels list 
        """
        if using == "pandas":
            return max(set(labels), key=labels.count)
        
        elif using == "numpy":
            count = {}
            for label in labels: 
                if label in count: 
                    count[label] += 1
                else:
                    count[label] = 1
    
            return max(count, key=count.get)
    
    def predict(self, X_test): 
        neighbours = []
        for datapoint in X_test:
            distances = self.distance(datapoint, self.X, metric="euclidean")
            top_index = np.argsort(distances)[:self.n_neighbors]
            neighbours.append(self.y[top_index])
        
        return list(map(self.most_common, neighbours))

    def evaluate(self, X_test, y_test):
        y_predicted = self.predict(X_test)
        accuracy = sum(y_predicted == y_test)/len(y_test)
        return accuracy
    
if __name__ == "__main__":
    x_train = np.random.randn(100, 2)
    y_train = np.random.randint(0, 5, 100)
    x_test = np.random.randn(10, 2)
    y_test = np.random.randint(0, 5, 10)

    knn = KNearstNeighbors(n_neighbors=5)
    knn.fit(x_train, y_train)
    knn.predict(x_test)