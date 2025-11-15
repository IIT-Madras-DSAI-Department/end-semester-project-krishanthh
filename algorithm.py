import pandas as pd
import numpy as np
from sklearn.metrics import f1_score

train = pd.read_csv('MNIST_train.csv')
val = pd.read_csv('MNIST_validation.csv')

Xtrain = train.iloc[:,1:-1].to_numpy()
ytrain = train.iloc[:,0].to_numpy().ravel().astype(int)
Xval = val.iloc[:,1:-1].to_numpy()
yval = val.iloc[:,0].to_numpy().ravel().astype(int)

class PCA:
    def __init__(self, n_components):
        if not isinstance(n_components, int) or n_components <= 0:
            raise ValueError("n_components must be a positive integer")
        self.n_components = n_components
        self.mean = None
        self.svalues_ = None        # columns are principal directions (n_features, n_components)

    def fit(self, X):
        if self.n_components == X.shape[1]:
            return X
        n_samples, n_features = X.shape
        self.mean_ = np.mean(X, axis=0)
        Xc = X - self.mean_
        U, S, Vt = np.linalg.svd(Xc, full_matrices=False)

        # Keep top n_components
        Vt_best = Vt[: self.n_components, :]         
        self.svalues_ = Vt_best.T
        return self

    def transform(self, X):
        if self.n_components == X.shape[1]:
            return X
        Xc = X - self.mean_
        return np.dot(Xc, self.svalues_)

    def inverse_transform(self, X_reduced):
        Xr = np.dot(X_reduced, self.svalues_.T) + self.mean_
        return Xr

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

class SoftmaxRegression:
    def __init__(self, learning_rate=0.1, epochs=100, n_components=784):
        self.learning_rate = learning_rate
        self.epochs = int(epochs)
        self.pca_model = PCA(n_components)
        self.W = None  
        self.b = None            

    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        expz = np.exp(z)
        return expz / np.sum(expz, axis=1, keepdims=True)
    
    def fit(self, X, y):
        self.pca_model.fit(X)
        X_reduced = self.pca_model.transform(X)
        n_samples_r, n_components = X.shape

        n_classes = (np.max(y)) + 1
        self.W = np.random.randn(n_components, n_classes) * 0.01
        self.b = np.zeros((1, n_classes))
        y_onehot = np.eye(n_classes)[y]        

        for epoch in range(self.epochs):
            logits = X.dot(self.W) + self.b
            probs = self._softmax(logits)
            grad_logits = (probs - y_onehot) / n_samples_r
            grad_W = X.T.dot(grad_logits)
            grad_b = np.sum(grad_logits, axis=0, keepdims=True)

            self.W -= self.learning_rate * grad_W
            self.b -= self.learning_rate * grad_b

            if epoch == self.epochs - 1:
                loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-15), axis=1))
                print(f"Epoch {epoch}: Loss = {loss:.6f}")

    def predict_proba(self, X):
        logits = X.dot(self.W) + self.b
        return self._softmax(logits)

    def predict(self, X):
        X_reduced = self.pca_model.transform(X)
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
class XGBoostTree:
    class Node:
        def __init__(self, feature_index=None, threshold=None, left=None, right=None, value=None):
            self.feature_index = feature_index
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
        
        def is_leaf_node(self):
            return self.value is not None
        
    def __init__(self, max_depth: int=3, min_samples_split: int=2, gamma: float=0.0, reg_lambda: float=1.0):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.gamma = gamma
        self.reg_lambda = reg_lambda
    
    def fit(self, X, g, h):
        self.n, self.m = X.shape
        self.root = self._build_tree(X, g, h, 0)

    def _build_tree(self, X, g, h, depth):
        if depth >= self.max_depth or X.shape[0] < self.min_samples_split:
            leaf_value = self._compute_leaf_value(g, h)
            return self.Node(value=leaf_value)
        
        best_feat, best_thresh = self._best_split(X, g, h)
        if best_feat is None or best_thresh is None:
            leaf_value = self._compute_leaf_value(g, h)
            return self.Node(value=leaf_value)

        left_mask = X[:, best_feat] <= best_thresh
        right_mask = ~left_mask

        child = self.Node(feature_index=best_feat, threshold=best_thresh,
                          left=self._build_tree(X[left_mask], g[left_mask], h[left_mask], depth+1),
                          right=self._build_tree(X[right_mask], g[right_mask], h[right_mask], depth+1))
        return child

    def _best_split(self, X, g, h):
        m, n = X.shape
        best_gain = -np.inf
        best_thresh, best_feat = None, None

        feature_indices = list(range(n))

        for feat_idx in feature_indices:
            X_col = X[:, feat_idx]
            sorted_idx = np.argsort(X[:, feat_idx])

            X_sorted, g_sorted, h_sorted = X_col[sorted_idx], g[sorted_idx], h[sorted_idx]

            G_prefix = np.cumsum(g_sorted)
            H_prefix = np.cumsum(h_sorted)
            G_total = G_prefix[-1]
            H_total = H_prefix[-1]

            for i in range(1, len(X_sorted)):
                if X_sorted[i] == X_sorted[i-1]:
                    continue

                GL, HL = G_prefix[i-1], H_prefix[i-1]
                GR, HR = G_total - GL, H_total - HL

                gain = 0.5 * (
                    (GL ** 2 / (HL + self.reg_lambda + 1e-8)) +
                    (GR ** 2 / (HR + self.reg_lambda + 1e-8)) -
                    (G_total ** 2 / (H_total + self.reg_lambda + 1e-8))
                ) - self.gamma

                if gain > best_gain and gain >= 0:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = (X_sorted[i] + X_sorted[i-1]) / 2

        return best_feat, best_thresh
    
    def _compute_leaf_value(self, g, h):
        G = np.sum(g)
        H = np.sum(h)
        return -G / (H + self.reg_lambda)
        
    def predict(self, X):
        X = np.array(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)
        
        preds = np.zeros(X.shape[0])
        for i in range(len(X)):
            node = self.root
            while not node.is_leaf_node():
                if X[i, node.feature_index] <= node.threshold:
                    node = node.left
                else:
                    node = node.right
            preds[i] = node.value
        return preds


class XGBoostClassifier:
    def __init__(self,n_estimators=50,learning_rate=0.1,max_depth=3,min_samples_split: int=2,gamma: float=0.0,reg_lambda: float=1.0,colsample=1.0,n_components=784,random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.gamma = gamma
        self.reg_lambda = reg_lambda
        self.colsample = colsample
        self.n_components = n_components
        self.pca_model = PCA(n_components)
        self.trees = {} 
        self.n_classes = None
        self.init_pred = None

        if random_state is not None:
            np.random.seed(random_state)

    def _softmax(self, z):
        z = z - np.max(z, axis=1, keepdims=True)
        expz = np.exp(z)
        return expz / np.sum(expz, axis=1, keepdims=True)

    def fit(self, X, y):
        self.pca_model.fit(X)
        X_reduced = self.pca_model.transform(X)
        
        X_reduced = np.array(X_reduced)
        y = np.array(y)
        
        
        self.n_classes = int(np.max(y)) + 1
        
        eps = 1e-15
        class_probs = np.bincount(y, minlength=self.n_classes) / len(y)
        class_probs = np.clip(class_probs, eps, 1 - eps)
        self.init_pred = np.log(class_probs)
        
        pred = np.tile(self.init_pred, (len(y), 1))
        
        for c in range(self.n_classes):
            self.trees[c] = []
        y_onehot = np.eye(self.n_classes)[y]
        

        for i in range(self.n_estimators):
            probs = self._softmax(pred)
            for c in range(self.n_classes):

                g = probs[:, c] - y_onehot[:, c]
                h = probs[:, c] * (1 - probs[:, c])
                n_features = int(self.colsample * X_reduced.shape[1])
                col_idx = np.random.choice(X_reduced.shape[1], n_features, replace=False)
                X_sub = X_reduced[:, col_idx]
                tree = XGBoostTree(
                    self.max_depth,
                    self.min_samples_split,
                    self.gamma,
                    self.reg_lambda
                )
                tree.fit(X_sub, g, h)
                
                pred[:, c] += self.learning_rate * tree.predict(X_reduced[:, col_idx])
                
                self.trees[c].append((tree, col_idx))
            
            if (i + 1) % 10 == 0 or i == self.n_estimators - 1:
                probs = self._softmax(pred)
                loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-15), axis=1))
                accuracy = np.mean(np.argmax(probs, axis=1) == y)
                print(f"Tree {i + 1}: Loss = {loss:.6f}, Accuracy = {accuracy:.4f}")

    def predict_proba(self, X):
        X_reduced = self.pca_model.transform(X)
        X_reduced = np.array(X_reduced)
        
        pred = np.tile(self.init_pred, (X_reduced.shape[0], 1))
        
        for c in range(self.n_classes):
            for tree, col_idx in self.trees[c]:
                pred[:, c] += self.learning_rate * tree.predict(X_reduced[:, col_idx])
        
        probs = self._softmax(pred)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)
    
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