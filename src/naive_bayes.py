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
            features (torch.Tensor): Bag of words representations of the training examples,
                                     shape = [num_samples, vocab_size].
            labels (torch.Tensor): Labels corresponding to each training example, shape = [num_samples].
            delta (float): Smoothing parameter for Laplace smoothing.
        """
        self.class_priors = self.estimate_class_priors(labels)
        self.vocab_size = features.shape[1]
        self.conditional_probabilities = self.estimate_conditional_probabilities(features, labels, delta)

    def estimate_class_priors(self, labels: torch.Tensor) -> Dict[int, torch.Tensor]:
        """
        Estimates class prior probabilities from the given labels.

        Args:
            labels (torch.Tensor): Labels corresponding to each training example.

        Returns:
            Dict[int, torch.Tensor]: A dictionary mapping class labels to their estimated prior probabilities.
        """
        N = labels.shape[0]
        class_priors: Dict[int, float] = {}
        for label in labels:
            label_item = int(label.item())
            if label_item not in class_priors:
                class_priors[label_item] = 0
            class_priors[label_item] += 1
        
        for c in class_priors:
            class_priors[c] = class_priors[c] / N
        
        class_priors = {c: torch.tensor(p, dtype=torch.float32) for c, p in class_priors.items()}
        return class_priors

    def estimate_conditional_probabilities(
        self, features: torch.Tensor, labels: torch.Tensor, delta: float
    ) -> Dict[int, torch.Tensor]:
        """
        Estimates conditional probabilities of words given a class using Laplace smoothing.

        Args:
            features (torch.Tensor): Bag of words representations of the training examples,
                                     shape = [num_samples, vocab_size].
            labels (torch.Tensor): Labels corresponding to each training example, shape = [num_samples].
            delta (float): Smoothing parameter for Laplace smoothing.

        Returns:
            Dict[int, torch.Tensor]: Conditional probabilities of each word for each class.
                                     e.g. { 0: [p(w1|0), p(w2|0), ...], 1: [p(w1|1), ...] }
        """
        vocab_size = features.shape[1]
        unique_classes = torch.unique(labels)
        
        class_word_counts: Dict[int, torch.Tensor] = {}

        for c in unique_classes:
            c_int = int(c.item())

            class_mask = (labels == c)
            class_sum = features[class_mask].sum(dim=0) 
            
            total_count_for_class = class_sum.sum()

            probabilities = (class_sum + delta) / (total_count_for_class + delta*vocab_size)
            
            class_word_counts[c_int] = probabilities
        
        return class_word_counts

    def estimate_class_posteriors(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Devuelve un vector con log P(y=c | x), para cada clase c.

        Args:
            feature (torch.Tensor): vector 1D de tamaño [vocab_size]
                                    con los conteos de cada palabra.

        Returns:
            torch.Tensor: vector 1D con log-probabilidades (no normalizadas) para cada clase.
        """
        if self.conditional_probabilities is None or self.class_priors is None:
            raise ValueError("El modelo debe estar entrenado antes de estimar los posteriors.")

        sorted_classes = sorted(self.class_priors.keys())

        log_posteriors = []
        for c in sorted_classes:
            log_prior = torch.log(self.class_priors[c])
            log_cond = torch.log(self.conditional_probabilities[c])

            log_likelihood = torch.sum(feature * log_cond)
            log_post = log_prior + log_likelihood
            log_posteriors.append(log_post)

        return torch.tensor(log_posteriors)

    def predict(self, feature: torch.Tensor) -> int:
        """
        Classifies a new feature using the trained Naive Bayes classifier.

        Args:
            feature (torch.Tensor): 1D vector [vocab_size] with word counts.

        Returns:
            int: The predicted class label (0 or 1 in binary classification).
        """
        log_post = self.predict_proba(feature)
        pred_class = torch.argmax(log_post).item()
        return pred_class

    def predict_proba(self, feature: torch.Tensor) -> torch.Tensor:
        """
        Predict the probability distribution over classes for a given feature vector.

        Args:
            feature (torch.Tensor): 1D vector [vocab_size] with word counts.

        Returns:
            torch.Tensor: A tensor representing the probability distribution (float) over all classes.
        """
        log_post = self.estimate_class_posteriors(feature)
        probs = torch.softmax(log_post, dim=0)
        return probs
