from MachineLearningModel import MachineLearningModel
import numpy as np

class LogisticRegressionModel(MachineLearningModel):
    # https://medium.com/analytics-vidhya/coding-logistic-regression-in-python-from-scratch-57284dcbfbff
    #https://medium.com/@stanweer/implementing-logistic-regression-algorithm-from-scratch-in-python-95b4d6874312
    def __init__(self, learning_rate=0.01, num_iterations=1000, epsilon= 1e-15):
            self.lr = learning_rate
            self.num_iterations = num_iterations
            self.epsilon = epsilon
            self.theta = np.ndarray
            self.cost_history = []

    def fit(self, X, y):
        """
        Train the logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.

        y (array-like): Target variable of the training data.


        Returns:

        None

        """

        X_arr, y_arr = self._check_shapes(X, y)
        m, n = X_arr.shape
        X_b = np.hstack([ np.ones((m,1)), X_arr ])
        self.theta = np.zeros(n + 1)
        self.cost_history = []
        #for gradient decent
        for _ in range(self.num_iterations):
                    z = X_b.dot(self.theta)
                    h = self._sigmoid(z)
                    grad = (1.0 / m) * X_b.T.dot(h - y_arr)
                    self.theta -= self.learning_rate * grad
                    cost = self._cost_function(X_b, y_arr)
                    self.cost_history.append(cost)
                                                                            
    def predict(self, X: np.ndarray, threshold: float = 0.5, return_proba = False):
            
        """
        Make predictions using the trained logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        #--- Write your code here ---#
        X_b = self._add_bias(X)
        proba = self._sigmoid(np.matmul(X_b, self.theta))
        if return_proba:
              return proba
        return (proba >= threshold).astype(int)
    

    def evaluate(self, X, y):
        """
        Evaluate the logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        score (float): Evaluation score (e.g., accuracy).
        """
        #--- Write your code here ---#
        x_arr, y_arr   = self._check_shapes(X, y)
        preds  = self.predict(x_arr)
        return np.mean(preds == y_arr)
    


    def _sigmoid(self, z):
        """
        Sigmoid function.

        Parameters:
        z (array-like): Input to the sigmoid function.

        Returns:
        result (array-like): Output of the sigmoid function.
        """
        #--- Write your code here ---#
        return 1.0 / (1.0 + np.exp(-z))

    def _cost_function(self, X_b, y_arr):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#
        m = y_arr.size
        z = X_b.dot(self.theta)
        h = self._sigmoid(z)
        #avoid log(0)
        h = np.clip(h, self.epsilon, 1-self.epsilon)
        cost = -(1.0 /m) * (y_arr.dot(np.log(h)) + (1 - y_arr).dot(np.log(1 - h)))
        return cost
    

    def add_bias(self, X_arr):
          m = X_arr.shape[0]
          return np.hstack([np.ones((m, 1), X_arr)])
    
    def _check_shapes(self, X, y):
        X_arr = np.array(X)
        y_arr = np.array(y).reshape(-1)
        if X_arr.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X_arr.shape}")
        if y_arr.ndim != 1:
            raise ValueError(f"y must be 1D shaped, got {y_arr.shape}")
        return X_arr, y_arr

