import numpy as np

class LogisticRgression(): 
    def __init__(self, learning_rate, iterations=50) -> None:
        self.learning_rate = learning_rate
        self.iterations = iterations
    
    def fit(self, X, y):
        self.X = X
        self.y = y
        self.n_samples, self.n_features = X.shape
        self.W = np.zeros(self.n_features)
        self.b = 0

        # apply gradient descent to update weights and bias 
        for _ in range(self.iterations): 
            self.update_weights()
        return self
    
    def update_weights(self): 
        z = np.dot(self.X, self.W) + self.b
        y_hat = self.sigmoid(z)
        # calculate gradients
        self.cost = self.loss(self.y, y_hat)
        dw = (1/self.n_samples) * np.dot(self.X.T, (y_hat - self.y))
        db = (1/self.n_samples) * np.sum(y_hat - self.y)
        # update weights and bias
        self.W -= self.learning_rate * dw 
        self.b -= self.learning_rate * db 
        
        return self 

    def loss(self, y, y_hat):
        loss = - np.mean(y * np.log(y_hat) + (1-y) * np.log(1-y_hat))   
        return loss 

    def predict(self, X):
        z = np.dot(X, self.W) + self.b
        y_hat = self.sigmoid(z)
        return y_hat
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    