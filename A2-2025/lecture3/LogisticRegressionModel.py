from MachineLearningModel import MachineLearningModel
import numpy as np
# https://medium.com/analytics-vidhya/coding-logistic-regression-in-python-from-scratch-57284dcbfbff
def __init__(self, learning_rate=0.01, num_iterations=1000, epsilon= 1e-15) -> None:
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.epsilon = epsilon
        self.theta = np.ndarray
        self.cost_history = []
        """
        Initialize the logistic regression model.

        Parameters:
        learning_rate (float): The learning rate for gradient descent.
        num_iterations (int): The number of iterations for gradient descent.
        """
        #--- Write your code here ---#

def fit(self, X, y):
    """
    Train the logistic regression model using gradient descent.

    Parameters:
    X (array-like): Features of the training data.

    y (array-like): Target variable of the training data.


    Returns:

    None

    """

    #--- Write your code here ---#

    X, y = self._check_shapes(X, y)
    m, n = X.shape
    X_b = np.hstack((np.ones((m + 1)), X))
    self.theta = np.zeros(n + 1)

    for _ in range(self.num_iterations):
        h    = self._sigmoid(np.matmul(X_b, self.theta))
        grad = (1.0 / m) * np.matmul(X_b.T, (h - y))
        self.theta -= self.learning_rate * grad
        self.cost_history.append(self._cost_function(X_b, y))

                                                                        
def predict(self, X: np.ndarray, threshold: float = 0.5, return_proba: bool = False) -> np.ndarray:
        
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
    return proba if return_proba else (proba >= threshold).astype(int)


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
    X, y   = self._check_shapes(X, y)
    preds  = self.predict(X)
    return (preds == y).mean()


def _sigmoid(self, z):
    """
    Sigmoid function.

    Parameters:
    z (array-like): Input to the sigmoid function.

    Returns:
    result (array-like): Output of the sigmoid function.
    """
    #--- Write your code here ---#

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