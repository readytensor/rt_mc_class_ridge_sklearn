import os
import warnings
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.exceptions import NotFittedError

warnings.filterwarnings("ignore")


PREDICTOR_FILE_NAME = "predictor.joblib"


class Classifier:
    """A wrapper class for the HistGradientBoosting multiclass classifier.

    This class provides a consistent interface that can be used with other
    classifier models.
    """

    model_name = "HistGradientBoosting_multiclass_classifier"

    def __init__(
            self,
            loss: Optional[str] = 'squared_error',
            learning_rate: Optional[float] = 0.1,
            max_depth: Optional[Union[int, None]] = None,
            max_leaf_nodes: Optional[Union[int, None]] = 31,
            min_samples_leaf: Optional[int] = 20,
            **kwargs,
    ):
        """Construct a new HistGradientBoosting classifier.

        Args:
            loss (optional, str):
            {‘squared_error’, ‘absolute_error’, ‘gamma’, ‘poisson’, ‘quantile’}, default=’squared_error’
            The loss function to use in the boosting process. Note that the “squared error”,
            “gamma” and “poisson” losses actually implement “half least squares loss”, “half gamma deviance” and “half
            poisson deviance” to simplify the computation of the gradient. Furthermore, “gamma” and “poisson” losses
            internally use a log-link, “gamma” requires y > 0 and “poisson” requires y >= 0. “quantile” uses the pinball
            loss.

            learning_rate (optional, float): The learning rate, also known as shrinkage. This is used as a
            multiplicative factor for the leaves values. Use 1 for no shrinkage.

            max_depth (optional, int, None): The maximum depth of each tree. The depth of a tree is the number of
            edges to go from the root to the deepest leaf. Depth isn’t constrained by default.

            max_leaf_nodes (optional, int, None): The maximum number of leaves for each tree. Must be strictly
            greater than 1. If None, there is no maximum limit.

            min_samples_leaf (optional, int): The minimum number of samples per leaf. For small datasets with less
            than a few hundred samples, it is recommended to lower this value since only very shallow trees would be
            built.

        """
        self.loss = loss
        self.learning_rate = float(learning_rate)
        self.max_depth = max_depth
        self.max_leaf_nodes = max_leaf_nodes
        self.min_samples_leaf = min_samples_leaf
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> HistGradientBoostingClassifier:
        """Build a new Random Forest binary classifier."""
        model = HistGradientBoostingClassifier(
            loss=self.loss,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            max_leaf_nodes=self.max_leaf_nodes,
            random_state=0
        )
        return model

    def fit(self, train_inputs: pd.DataFrame, train_targets: pd.Series) -> None:
        """Fit the HistGradientBoosting multiclass classifier to the training data.

        Args:
            train_inputs (pandas.DataFrame): The features of the training data.
            train_targets (pandas.Series): The labels of the training data.
        """
        self.model.fit(train_inputs, train_targets)
        self._is_trained = True

    def predict(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class labels for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class labels.
        """
        return self.model.predict(inputs)

    def predict_proba(self, inputs: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities for the given data.

        Args:
            inputs (pandas.DataFrame): The input data.
        Returns:
            numpy.ndarray: The predicted class probabilities.
        """

        return self.model.predict_proba(inputs)

    def evaluate(self, test_inputs: pd.DataFrame, test_targets: pd.Series) -> float:
        """Evaluate the HistGradientBoosting binary classifier and return the accuracy.

        Args:
            test_inputs (pandas.DataFrame): The features of the test data.
            test_targets (pandas.Series): The labels of the test data.
        Returns:
            float: The accuracy of the HistGradientBoosting binary classifier.
        """
        if self.model is not None:
            return self.model.score(test_inputs, test_targets)
        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the HistGradientBoosting binary classifier to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "Classifier":
        """Load the HistGradientBoosting binary classifier from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            Classifier: A new instance of the loaded HistGradientBoosting binary classifier.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model

    def __str__(self):
        # sort params alphabetically for unit test to run successfully
        return (
            f"Model name: {self.model_name} ("
            f"learning_rate: {self.learning_rate}, "
            f"loss: {self.loss}, "
            f"max_depth: {self.max_depth}, "
            f"max_leaf_nodes: {self.max_leaf_nodes}, "
            f"min_samples_leaf: {self.min_samples_leaf})"
        )


def train_predictor_model(
    train_inputs: pd.DataFrame, train_targets: pd.Series, hyperparameters: dict
) -> Classifier:
    """
    Instantiate and train the predictor model.

    Args:
        train_inputs (pd.DataFrame): The training data inputs.
        train_targets (pd.Series): The training data labels.
        hyperparameters (dict): Hyperparameters for the classifier.

    Returns:
        'Classifier': The classifier model
    """
    classifier = Classifier(**hyperparameters)
    classifier.fit(train_inputs=train_inputs, train_targets=train_targets)
    return classifier


def predict_with_model(
    classifier: Classifier, data: pd.DataFrame, return_probs=False
) -> np.ndarray:
    """
    Predict class probabilities for the given data.

    Args:
        classifier (Classifier): The classifier model.
        data (pd.DataFrame): The input data.
        return_probs (bool): Whether to return class probabilities or labels.
            Defaults to True.

    Returns:
        np.ndarray: The predicted classes or class probabilities.
    """
    if return_probs:
        return classifier.predict_proba(data)
    return classifier.predict(data)


def save_predictor_model(model: Classifier, predictor_dir_path: str) -> None:
    """
    Save the classifier model to disk.

    Args:
        model (Classifier): The classifier model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> Classifier:
    """
    Load the classifier model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        Classifier: A new instance of the loaded classifier model.
    """
    return Classifier.load(predictor_dir_path)


def evaluate_predictor_model(
    model: Classifier, x_test: pd.DataFrame, y_test: pd.Series
) -> float:
    """
    Evaluate the classifier model and return the accuracy.

    Args:
        model (Classifier): The classifier model.
        x_test (pd.DataFrame): The features of the test data.
        y_test (pd.Series): The labels of the test data.

    Returns:
        float: The accuracy of the classifier model.
    """
    return model.evaluate(x_test, y_test)
