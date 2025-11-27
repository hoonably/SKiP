import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted
from sklearn.preprocessing import LabelEncoder


class OneVsRestSVM(BaseEstimator, ClassifierMixin):
    """
    One-vs-Rest (OvR) multiclass SVM classifier.
    
    Wraps binary SVM classifiers (NaiveSVM, ProbSVM, KNNSVM, SKiP) 
    to handle multiclass classification using the One-vs-Rest strategy.
    
    Parameters:
        estimator: Binary SVM classifier instance (e.g., NaiveSVM, ProbSVM, KNNSVM, SKiP)
    """
    
    def __init__(self, estimator):
        self.estimator = estimator
    
    def fit(self, X, y):
        """
        Fit one binary classifier per class.
        
        Parameters:
            X: Training data, shape (n_samples, n_features)
            y: Target labels, shape (n_samples,)
        
        Returns:
            self
        """
        X, y = check_X_y(X, y, accept_sparse=False, dtype=np.float64, order='C')
        
        self._label_encoder = LabelEncoder()
        y_encoded = self._label_encoder.fit_transform(y)
        self.classes_ = self._label_encoder.classes_
        
        n_classes = len(self.classes_)
        self.estimators_ = []
        
        # Train one classifier per class
        for i in range(n_classes):
            # Create binary labels: +1 for current class, -1 for rest
            y_binary = np.where(y_encoded == i, 1, 0)
            
            # Clone estimator with same parameters
            from sklearn.base import clone
            clf = clone(self.estimator)
            clf.fit(X, y_binary)
            
            self.estimators_.append(clf)
        
        return self
    
    def decision_function(self, X):
        """
        Compute decision function for all classes.
        
        Parameters:
            X: Test data, shape (n_samples, n_features)
        
        Returns:
            Decision values, shape (n_samples, n_classes)
        """
        check_is_fitted(self, attributes=["estimators_", "classes_"])
        X = check_array(X, accept_sparse=False, dtype=np.float64, order='C')
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        
        # Collect decision values from all binary classifiers
        decision_values = np.zeros((n_samples, n_classes))
        
        for i, clf in enumerate(self.estimators_):
            decision_values[:, i] = clf.decision_function(X)
        
        return decision_values
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
            X: Test data, shape (n_samples, n_features)
        
        Returns:
            Predicted labels
        """
        decision_values = self.decision_function(X)
        
        # Predict the class with highest decision value
        y_pred_encoded = np.argmax(decision_values, axis=1)
        
        # Convert back to original labels
        return self._label_encoder.inverse_transform(y_pred_encoded)
