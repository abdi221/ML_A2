import numpy as np

class ROCAnalysis:
    """
    Class to calculate various metrics for Receiver Operating Characteristic (ROC) analysis.

    Attributes:
        y_pred (list): Predicted labels.
        y_true (list): True labels.
        tp (int): Number of true positives.
        tn (int): Number of true negatives.
        fp (int): Number of false positives.
        fn (int): Number of false negatives.
    """

    def __init__(self, y_predicted, y_true):
        """
        Initialize ROCAnalysis object.

        Parameters:
            y_predicted (list): Predicted labels (0 or 1).
            y_true (list): True labels (0 or 1).
        """
        self.y_pred = np.asarray(y_predicted, dtype=int).ravel()
        self.y_true = np.asarray(y_true, dtype=int).ravel()

        if self.y_pred.shape != self.y_true.shape:
            raise ValueError(f"predicted and true must have the same shape, \n" 
                             f"got {self.y_pred.shape} vs {self.y_true.shape}")
        
        #confusion matrix entries
        self.tp = int(np.sum((self.y_pred == 1) & (self.y_true == 1)))
        self.tp = int(np.sum((self.y_pred == 0) & (self.y_true == 0)))
        self.tp = int(np.sum((self.y_pred == 1) & (self.y_true == 0)))
        self.tp = int(np.sum((self.y_pred == 0) & (self.y_true == 1)))

    def tp_rate(self):
        """
        Calculate True Positive Rate (Sensitivity, Recall).

        Returns:
            float: True Positive Rate.
        """
        #--- Write your code here ---#
        denom = self.tp + self.fn
        return self.tp/denom if denom > 0 else 0.0
    

    def fp_rate(self):
        """
        Calculate False Positive Rate.

        Returns:
            float: False Positive Rate.
        """
        #--- Write your code here ---#
        denom = self.fp + self.tn
        return self.fp / denom if denom > 0 else 0.0
    
    def precision(self):
        """
        Calculate Precision.

        Returns:
            float: Precision.
        """
        #--- Write your code here ---#
        denom = self.tp + self.fp
        return self.tp / denom if denom > 0 else 0.0
  
    def f_score(self, beta=1):
        """
        Calculate the F-score.

        Parameters:
            beta (float, optional): Weighting factor for precision in the harmonic mean. Defaults to 1.

        Returns:
            float: F-score.
        """
        #--- Write your code here ---#
