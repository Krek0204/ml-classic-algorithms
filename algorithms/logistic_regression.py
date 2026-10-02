import numpy as np


class LogisticRegression:
    def __init__(self, lr=0.001, num_epochs = 200, tr=0.5):
        self.lr = lr
        self.num_epochs = num_epochs
        self.tr = tr
        
    def _sigmoid(self, z):
        z = np.clip(z, -500, 500)
        return 1 / (1 + np.exp(-z))
    
    def change_threshold(self, threshold):
        self.tr = threshold
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        # Берём размерность весов на 1 больше, поскольку в weights включаем также bias
        self.weights = np.zeros(n_features + 1)
        
        for _ in range(self.num_epochs):
            # Случайная перестановка для стохастического градиентного спуска
            indices = np.random.permutation(n_samples)
            
            for i  in indices:
                xi = np.array(X[i])
                xi = np.insert(xi, 0, 1)
                yi = y[i]
                
                logit = np.dot(xi, self.weights)
                predict = self._sigmoid(logit)
                grad = (predict - yi) * xi
                
                # Обновляем веса
                self.weights = self.weights  - self.lr * grad
                
                
        
    def predict(self, X):
        logits = np.dot(X, self.weights[1:]) + self.weights[0]
        probs = self._sigmoid(logits)
        results = [1 if prob >= self.tr else 0 for prob in probs]
        return results