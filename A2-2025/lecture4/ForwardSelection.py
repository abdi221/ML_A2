import numpy as np
from lecture4.ROCAnalysis import ROCAnalysis

class ForwardSelection:
    """
    A class for performing forward feature selection based on maximizing the F-score of a given model.

    Attributes:
        X (array-like): Feature matrix.
        y (array-like): Target labels.
        model (object): Machine learning model with `fit` and `predict` methods.
        selected_features (list): List of selected feature indices.
        best_cost (float): Best F-score achieved during feature selection.
    """

    def __init__(self, X, y, model, random_state=None):
        """
        Initializes the ForwardSelection object.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.
            model (object): Machine learning model with `fit` and `predict` methods.
        """
        #--- Write your code here ---#
        self.X  = X
        self.y = y
        self.model = model
        self.n_samples, self.n_features = X.shape

        #to keep track of selection
        self.selected_features = []
        self.best_score = 0.0

        #creat eone internal split for all evaluations
        self.X_train, self.X_tst, self.y_train, self.y_tst = self.create_split(X, y, random_state=None)



    def create_split(self, X, y, random_state=None):
        """
        Creates a train-test split of the data.

        Parameters:
            X (array-like): Feature matrix.
            y (array-like): Target labels.

        Returns:
            X_train (array-like): Features for training.
            X_test (array-like): Features for testing.
            y_train (array-like): Target labels for training.
            y_test (array-like): Target labels for testing.
        """
        #--- Write your code here ---#
        rng = np.random.RandomState(random_state)
        idx = rng.permutation(self.n_samples)
        cut = int(0.8*self.n_samples)
        train_idx, tst_idx = idx[:cut], idx[cut:]
        return X[train_idx], X[tst_idx], y[train_idx], y[tst_idx]
    

    def train_model_with_features(self, features):
        """
        Trains the model using selected features and evaluates it using ROCAnalysis.

        Parameters:
            features (list): List of feature indices.

        Returns:
            float: F-score obtained by evaluating the model.
        """
        #--- Write your code here ---#

    def forward_selection(self):
        """
        Performs forward feature selection based on maximizing the F-score.
        """
        #--- Write your code here ---#
                
    def fit(self):
        """
        Fits the model using the selected features.
        """
        #--- Write your code here ---#

    def predict(self, X_test):
        """
        Predicts the target labels for the given test features.

        Parameters:
            X_test (array-like): Test features.

        Returns:
            array-like: Predicted target labels.
        """
        #--- Write your code here ---#
