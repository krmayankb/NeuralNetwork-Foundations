import numpy as np

class LinearRgression(): 
    def __init__(self, learning_rate, iterations=100):
        """
        Custom implementation of Linear Regression using Gradient Descent 

        Parameters
        ----------
        learning_rate : float
            The learning rate for the gradient descent algorithm
        iterations : int
            The number of iterations to run the gradient descent algorithm
        """
        self.learning_rate = learning_rate
        self.iterations = iterations
    
    def fit(self, X, y):
        """
        Fit the linear regression model to the data

        Parameters
        ----------
        X : numpy.ndarray
            The input data of shape (m, n)
        y : numpy.ndarray
            The given labels for the input data
        """
        self.m, self.n = X.shape 
        
        # initialize weights and bias
        self.W = np.zeros(self.n)
        self.b = 0 
        self.X = X
        self.y = y

        # apply gradient descent to update weights and bias 
        for _ in range(self.iterations): 
            self.update_weights()
        return self
    
    def update_weights(self): 
        predicted_y = self.predict(self.X)
        # calculate gradients
        dw = -(2/self.m) * np.dot(self.X.T, (self.y - predicted_y))
        db = -(2/self.m) * np.sum(self.y - predicted_y)

        self.W  -= self.learning_rate * dw 
        self.b -= self.learning_rate * db
        return self 

    def predict(self, X):
        return np.dot(X, self.W) + self.b 
    
    def loss(self, y, y_pred):
        return np.mean((y - y_pred)**2)
