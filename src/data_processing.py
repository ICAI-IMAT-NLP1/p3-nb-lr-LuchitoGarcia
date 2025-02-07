from typing import List, Dict
from collections import Counter
import torch

try:
    from src.utils import SentimentExample, tokenize
except ImportError:
    from utils import SentimentExample, tokenize


def read_sentiment_examples(infile: str) -> List[SentimentExample]:
    """
    Reads sentiment examples from a file.

    Args:
        infile: Path to the file to read from.

    Returns:
        A list of SentimentExample objects parsed from the file.
    """
    with open(infile, "r") as f:
        lines = f.readlines()
        examples: List[SentimentExample] = []
        for line in lines:
            words_list = line.rstrip().replace("\t", " ").split(" ")
            object = SentimentExample(words_list[0:-1], int(words_list[-1]))
            examples.append(object)
        
    return examples

def build_vocab(examples: List[SentimentExample]) -> Dict[str, int]:
    """
    Creates a vocabulary from a list of SentimentExample objects.

    The vocabulary is a dictionary where keys are unique words from the examples and values are their corresponding indices.

    Args:
        examples (List[SentimentExample]): A list of SentimentExample objects.

    Returns:
        Dict[str, int]: A dictionary representing the vocabulary, where each word is mapped to a unique index.
    """
    vocab: Dict[str, int] = {}
    counter: int = 0
    for example in examples:
        for word in example.words:
            if word not in vocab:
                vocab[word] = counter
                counter += 1
    return vocab


def bag_of_words(
    text: List[str], vocab: Dict[str, int], binary: bool = False
) -> torch.Tensor:
    """
    Converts a list of words into a bag-of-words vector based on the provided vocabulary.
    Supports both binary and full (frequency-based) bag-of-words representations.

    Args:
        text (List[str]): A list of words to be vectorized.
        vocab (Dict[str, int]): A dictionary representing the vocabulary with words as keys and indices as values.
        binary (bool): If True, use binary BoW representation; otherwise, use full BoW representation.

    Returns:
        torch.Tensor: A tensor representing the bag-of-words vector.
    """
    # TODO: Converts list of words into BoW, take into account the binary vs full
    bow: torch.Tensor = torch.zeros(size = (len(vocab),), dtype=torch.float32)
    for word in text:
        if word in vocab:
            if not binary:
                bow[vocab[word]] += 1
            else:
                bow[vocab[word]] = 1
    return bow
