from nltk.lm.preprocessing import padded_everygram_pipeline
import numpy as np
import math
from collections import defaultdict, Counter


# Compute the perplexity of the My Stupid Backoff model
def compute_ppl_my_sb(model, data):
    highest_ngram = max(model.ngrams.keys())
    scores = []

    for sentence in data:
        # if sentence is type of list object
        if isinstance(sentence, list):
            ngrams, flat_text = padded_everygram_pipeline(highest_ngram, [sentence])
        else:
            ngrams, flat_text = padded_everygram_pipeline(highest_ngram, [sentence.split()])

        scores.extend([-1 * model.logscore(w[-1], w[0:-1]) for gen in ngrams for w in gen if len(w) == highest_ngram])

    return math.pow(2.0, np.asarray(scores).mean())


# compute perplexity of the model
def compute_ppl_(model, data):
    highest_ngram = model.order
    scores = []

    for sentence in data:
        # if sentence is type of list object
        if isinstance(sentence, list):
            ngrams, flat_text = padded_everygram_pipeline(highest_ngram, [sentence])
        else:
            ngrams, flat_text = padded_everygram_pipeline(highest_ngram, [sentence.split()])

        scores.extend([-1 * model.logscore(w[-1], w[0:-1]) for gen in ngrams for w in gen if len(w) == highest_ngram])

    return math.pow(2.0, np.asarray(scores).mean())


class MyStupidBackoff:
    def __init__(self, corpus, n, alpha=0.4):
        """
        Initialize the MyStupidBackoff language model.

        Args:
            corpus (list): The training corpus as a list of words.
            n (int): The order of the n-gram model.
            alpha (float): The backoff factor, defaults to 0.4.
        """
        self.alpha = alpha
        self.ngrams = defaultdict(lambda: defaultdict(Counter))  # Dictionary to store n-gram counts
        self.N = len(corpus)  # Size of the training corpus
        self.build_ngrams(corpus, n)  # Build the n-gram model

    def build_ngrams(self, corpus, n):
        """
        Build the n-gram model by counting n-grams and their frequencies.

        Args:
            corpus (list): The training corpus as a list of words.
            n (int): The order of the n-gram model.
        """
        for i in range(n-1, len(corpus)):
            for j in range(i-n+1, i):
                prefix = tuple(corpus[j:i])  # Context prefix of length n-1
                token = corpus[i]  # Current word (target token)
                self.ngrams[len(prefix)][prefix][token] += 1  # Increment n-gram count

    def score(self, word, context):
        """
        Compute the Stupid Backoff score for a word given a context.

        Args:
            word (str): The target word.
            context (list): The context as a list of words.

        Returns:
            float: The Stupid Backoff score.
        """
        context = tuple(context)
        if context in self.ngrams[len(context)] and word in self.ngrams[len(context)][context]:
            return self.ngrams[len(context)][context][word] / sum(self.ngrams[len(context)][context].values())
        elif len(context) > 0:
            return self.alpha * self.score(word, context[1:])
        else:
            return 1e-2  # Return a small probability for unseen words

    def logscore(self, word, context):
        """
        Compute the logarithm of the Stupid Backoff score for a word given a context.

        Args:
            word (str): The target word.
            context (list): The context as a list of words.

        Returns:
            float: The logarithm of the Stupid Backoff score.
        """
        return math.log(self.score(word, context))
