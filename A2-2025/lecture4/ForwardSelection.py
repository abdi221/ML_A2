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
        #subset feattures 
        Xtr = self.X_train[:, features]
        Xte = self.X_tst[:, features]

        #fit and predct 
        self.model.fit(Xtr, self.y_train)
        y_pred = self.model.predict(Xte)
        
        # evaluate
        roc = ROCAnalysis(self.y_tst, y_pred)
        return roc.f_score()

    def forward_selection(self):
        """
        Performs forward feature selection based on maximizing the F-score.
        """
        #--- Write your code here ---#
        remaining = set(range(self.n_features))

        #start from no features 
        while remaining:
            best_feat = None
            best_score = self.best_cost


            #try adding each remaining feat
            for feat in remaining:
                trial = self.selected_features + [feat]
                score = self.train_model_with_features(trial)
                if score > best_score:
                    best_score = score
                    best_feat = feat
            
            #if it finds an improvement, it should fix that feature and continue
            if best_feat is not None:
                self.selected_features.append(best_feat)
                remaining.remove(best_feat)
                self.best_score = best_score
            else:
                break # no further gain
                
    def fit(self):
        """
        Fits the model using the selected features.
        """
        #--- Write your code here ---#

        #find the best subset
        self.forward_selection()

        # final fit on full data
        X_sel = self.X[:, self.selected_features]
        self.model.fit(X_sel, self.y)

    def predict(self, X_test):
        """
        Predicts the target labels for the given test features.

        Parameters:
            X_test (array-like): Test features.

        Returns:
            array-like: Predicted target labels.
        """
        #--- Write your code here ---#

        X_sel = X_test[:, self.selected_features]
        return self.model.predict(X_sel)
