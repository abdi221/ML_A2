import numpy as np

class NumpyLogistic:
    """
    Binary logistic regression via batch gradient descent.
    Implements .fit(X, y) and .predict(X) for compatibility.
    """
    def __init__(self, lr=0.1, n_iter=1000, tol=1e-6, verbose=False):
        self.lr      = lr
        self.n_iter  = n_iter
        self.tol     = tol
        self.verbose = verbose

    def _sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0.0

        for i in range(self.n_iter):
            z      = X.dot(self.w) + self.b
            y_pred = self._sigmoid(z)

            error   = y_pred - y
            grad_w  = (X.T @ error) / n_samples
            grad_b  = np.mean(error)

            self.w -= self.lr * grad_w
            self.b -= self.lr * grad_b

            grad_norm = np.linalg.norm(np.hstack((grad_w, grad_b)))
            if grad_norm < self.tol:
                if self.verbose:
                    print(f"Converged after {i+1} iterations")
                break

        return self

    def predict_proba(self, X):
        return self._sigmoid(X.dot(self.w) + self.b)

    def predict(self, X):
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
