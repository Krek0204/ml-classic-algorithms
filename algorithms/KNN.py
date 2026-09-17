import numpy as np

class KNN:
    def __init__(self, n_neighbors=5, regression=False):
        self.n_neighbors = n_neighbors
        self.regression = regression
        
    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train
        
    def _euclidian_distances(self, x_test_i):
        return np.sqrt(np.sum((self.X_train - x_test_i) ** 2, axis=1))
    
    def _make_prediction(self, x_test_i):
        distances = self._euclidian_distances(x_test_i)
        k_nearest_indexes = np.argsort(distances)[:self.n_neighbors]
        targets = self.y_train[k_nearest_indexes]
        
        return np.mean(targets) if self.regression else np.bincount(targets).argmax()
    
    def predict(self, X_test):
        return np.array([self._make_prediction(x_test_i) for x_test_i in X_test])   
    
         
        