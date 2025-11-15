import pandas as pd
import numpy as np
import sys
from sklearn.metrics import f1_score

train = pd.read_csv('MNIST_train.csv')
Xtrain = train.iloc[:,1:-1].to_numpy()
ytrain = train.iloc[:,0].to_numpy().ravel().astype(int)

class KNNClassifier:
    def __init__(self,k=5,distance_metric='euclidean',X=Xtrain,y=ytrain):
        self.k = k
        self.distance_metric = distance_metric
        self.X_train = X
        self.y_train = y
        self.n_classes = np.max(y)+1
    
    def _compute_distance(self, x1, x2):
        if self.distance_metric == 'euclidean':
            return np.sqrt(np.sum((x1 - x2) ** 2, axis=1))
        elif self.distance_metric == 'manhattan':
            return np.sum(np.abs(x1 - x2), axis=1)
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")

    
    def _predict_single(self, x):
        distances = self._compute_distance(self.X_train, x)
        k_nearest_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = self.y_train[k_nearest_indices]
        
        # Majority vote
        label_counts = np.bincount(k_nearest_labels, minlength=self.n_classes)
        predicted_label = np.argmax(label_counts)
        
        return predicted_label, label_counts
    
    def predict(self, X):
        X = np.array(X)
        # Transform using fitted PCA
        
        predictions = np.zeros(len(X), dtype=int)
        
        print(f"[KNN] Predicting {len(X)} samples...")
        for i in range(len(X)):
            predictions[i], _ = self._predict_single(X[i:i+1])
            
            if (i + 1) % 100 == 0 or i == len(X) - 1:
                print(f"[KNN] Processed {i + 1}/{len(X)} samples")
        return predictions
    
if __name__ == "__main__":
    test_path = sys.argv[1]
    test = pd.read_csv(test_path)
    
    Xtest = test.iloc[:, 1:-1].to_numpy()
    
    ytest = test.iloc[:, 0].to_numpy().ravel().astype(int)
    
    knn = KNNClassifier(k=5, distance_metric='euclidean')
    y_pred = knn.predict(Xtest)
    
    print(f"F1 Score: {f1_score(y_pred, ytest, average='weighted'):.4f}")
    print(y_pred)