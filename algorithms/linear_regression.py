import numpy as np


class LinearRegression:
    def __init__(self, lr = 0.01, num_epoch = 500):
        self.learning_rate = lr
        self.num_epoch = 500
        
    def fit(self, X, y) -> None:
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features + 1)
        
        for _ in range(self.num_epoch):
            # Случайная перестановка для стохастического градиентного спуска
            indices = np.random.permutation(n_samples)
            
            for i in indices:
                xi = np.array(X[i])
                xi = np.insert(xi, 0, 1)
                yi = np.array(y[i])
                
                grad = 2 * xi.T.dot(xi.dot(self.weights) - yi)
                self.weights = self.weights - self.learning_rate * grad
                
    def predict(self, x) -> int:
        return np.dot(self.weights[1:], x) + self.weights[0]
                