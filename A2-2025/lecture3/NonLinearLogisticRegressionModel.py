from MachineLearningModel import MachineLearningModel
import numpy as np

"""
Nonlinear Logistic Regression model using gradient descent optimization.
It works for 2 features (when creating the variable interactions)
"""

class NonLinearLogisticRegressionModel(MachineLearningModel):
    def __init__(self, degree=2, learning_rate=0.01, num_iterations=1000, epsilon = 1e-15):
        """
        Initialize the nonlinear logistic regression model.

        Parameters:
        degree (int): Degree of polynomial features.
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        #--- Write your code here ---#
        self.degree = degree
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.theta = None
        self.epsilon = epsilon
        self.cost_history = []

    def fit(self, X, y):
        """
        Train the nonlinear logistic regression model using gradient descent.

        Parameters:
        X (array-like): Features of the training data.
        y (array-like): Target variable of the training data.

        Returns:
        None
        """
        #--- Write your code here ---#
        X_arr = np.array(X)
        y_arr = np.array(y).reshape(-1)
        if X_arr.ndim != 2 or X_arr.shape[1] != 2:
            raise ValueError(f"X must be (m, 2), got {X_arr.shape}")
        x1, x2 = X_arr[:, 0], X_arr[:, 1]
        X_mapped = self.mapFeature(x1, x2)
        m, _ = X_mapped.shape
        self.theta = np.zeros(X_mapped.shape[1])
        self.cost_history = []

        for _ in range(self.num_iterations):
            z = X_mapped.dot(self.theta)
            h = self._sigmoid(z)
            grad = (1.0 / m) * X_mapped.T.dot(h - y_arr)
            self.theta -= self.learning_rate * grad
            cost = self._cost_function(X_mapped, y_arr)
            self.cost_history.append(cost)


    def predict(self, X, threshold=0.5, return_probe=False):
        """
        Make predictions using the trained nonlinear logistic regression model.

        Parameters:
        X (array-like): Features of the new data.

        Returns:
        predictions (array-like): Predicted probabilities.
        """
        #--- Write your code here ---#
        X_arr = np.array(X)
        if X_arr.ndim !=2 or X_arr.shape[1] != 2:
            raise ValueError(f"X must be (m,2), got {X_arr.shape}")
        X_mapped = self.mapFeature(X_arr[:, 0], X_arr[:, 1])
        proba = self._sigmoid(X_mapped.dot(self.theta))
        if return_probe:
            return proba
        return (proba >= threshold).astype(int)
    


    def evaluate(self, X, y):
        """
        Evaluate the nonlinear logistic regression model on the given data.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#
        preds = self.predict(X)
        y_arr = np.array(y).reshape(-1) 
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
    #https://numpy.org/doc/2.2/reference/generated/numpy.vstack.html
    def mapFeature(self, X1, X2):
        """
        Map the features to a higher-dimensional space using polynomial features.
        Check the slides to have hints on how to implement this function.
        Parameters:
        X1 (array-like): Feature 1.
        X2 (array-like): Feature 2.
        D (int): Degree of polynomial features.

        Returns:
        X_poly (array-like): Polynomial features.
        """
        #--- Write your code here ---#
        m = X1.shape[0]
        features = [np.ones(m)]
        for i in range(1, self.degree + 1):
            for j in range(i +1):
                features.append((X1 ** (i - j)) * (X2 **j))
        return np.vstack(features).T
    
    def _cost_function(self, X, y):
        """
        Compute the logistic regression cost function.

        Parameters:
        X (array-like): Features of the data.
        y (array-like): Target variable of the data.

        Returns:
        cost (float): The logistic regression cost.
        """
        #--- Write your code here ---#
        m = y.size
        z = X.dot(self.theta)
        h = self._sigmoid(z)
        h = np.clip(h, self.epsilon, 1-self.epsilon)
        return -(1.0/m) * (y.dot(np.log(h)) + (1 -y).dot(np.log(1 -h)))
    