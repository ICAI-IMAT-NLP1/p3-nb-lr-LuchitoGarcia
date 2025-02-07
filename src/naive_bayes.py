import torch
from collections import Counter
from typing import Dict

try:
    from src.utils import SentimentExample
    from src.data_processing import bag_of_words
except ImportError:
    from utils import SentimentExample
    from data_processing import bag_of_words


class NaiveBayes:
    def __init__(self):
        """
        Initializes the Naive Bayes classifier
        """
        self.class_priors: Dict[int, torch.Tensor] = None
        self.conditional_probabilities: Dict[int, torch.Tensor] = None
        self.vocab_size: int = None

    def fit(self, features: torch.Tensor, labels: torch.Tensor, delta: float = 1.0):
        """
        Trains the Naive Bayes classifier by initializing class priors and estimating conditional probabilities.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        # TODO: Estimate class priors and conditional probabilities of the bag of words 
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = features.shape[1] # Shape of the probability tensors, useful for predictions and conditional probabilities
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)
        return

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """

        N: int = labels.shape[0]
        class_priors: Dict[int, torch.Tensor] = {}
        for i in range(N):
            label = labels[i].item()
            if label not in class_priors:
                class_priors[label] = 1 / N
            else:
                class_priors[label] += 1 / N
        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples.
            labels (torch.Tensor): Labels corresponding to each training example.
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
        """
        class_word_counts: Dict[int, torch.Tensor] = {}
        total_word_count: int = 0
        total_word_count += features.sum().item()

        for i in range(features.shape[0]):
            feature = features[i]
            label = labels[i].item()

            if label not in class_word_counts:
                    class_word_counts[label] = torch.zeros(size=(features.shape[1],),dtype=torch.float32)

            for j in range(features.shape[1]):
                class_word_counts[label][j] = (delta + feature[j]) / (delta + total_word_count)
        
        return class_word_counts

    def estimate_class_posteriors(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Calcula el log posterior para cada clase dada una sola bolsa de palabras `feature`.

        Args:
            feature (torch.Tensor): vector 1D de tamaño [vocab_size] 
                que contiene las frecuencias (o conteos) de cada palabra.

        Returns:
            torch.Tensor: vector 1D con el log posterior de cada clase.
        """
        # Asegurarse de que el modelo esté entrenado
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError("El modelo debe estar entrenado antes de estimar los posteriors.")

        log_posteriors = torch.zeros(len(self.class_priors), dtype=torch.float32)
        for i in range(len(self.conditional_probabilities)):
            p_x = sum([self.conditional_probabilities[i]@feature * self.class_priors[i]])
        for j in range(len(log_posteriors)):
            log_posteriors[j] = torch.log(self.conditional_probabilities[j]@feature * self.class_priors[j] / p_x)

        return log_posteriors

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): The  feature vector (bag of words representation) of the example to classify.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")
        
        pred: int = torch.argmax(self.estimate_class_posteriors(feature)).item()
        return pred

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): The feature vector (bag of words representation) of the example.

        Returns:
            torch.Tensor: A tensor representing the probability distribution over all classes.

        Raises:
            Exception: If the model has not been trained before calling this method.
        """
        if not self.class_priors or not self.conditional_probabilities:
            raise Exception("Model not trained. Please call the train method first.")

        probs: torch.Tensor = torch.softmax(self.estimate_class_posteriors(feature), dim=0)
        return probs 