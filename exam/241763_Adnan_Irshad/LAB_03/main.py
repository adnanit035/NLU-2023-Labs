from nltk.util import everygrams
from nltk.lm.preprocessing import flatten
from nltk.lm.preprocessing import padded_everygram_pipeline
from nltk.lm import Vocabulary
from nltk.corpus import gutenberg
from nltk.lm import StupidBackoff

from functions import *


if __name__ == "__main__":
    # Load data
    macbeth_sents = [[w.lower() for w in sent] for sent in gutenberg.sents('shakespeare-Macbeth.txt')]
    macbeth_words = [w.lower() for w in gutenberg.words('shakespeare-Macbeth.txt')]

    # Vocabulary and n-grams
    n = 2  # n-gram order

    # computing vocabulary with cutoff
    vocab = Vocabulary(macbeth_words, unk_cutoff=2)

    # extracting ngrams
    padded_ngrams_oov, flat_text_oov = padded_everygram_pipeline(n, macbeth_sents)

    ##########################################################################################################
    # 1. NLTK's Stupid Backoff Model Training and Testing
    ##########################################################################################################
    # Train the Stupid Backoff model
    alpha = 0.4
    model_sb = StupidBackoff(alpha=alpha, order=n, vocabulary=vocab)
    model_sb.fit(padded_ngrams_oov, flat_text_oov)

    # Test the Stupid Backoff model
    test_sents1 = ["the king is dead", "the emperor is dead", "may the force be with you"]
    test_sents2 = ["the king is dead", "welcome to you", "how are you"]

    print("\nStupid Backoff Model with alpha = " + str(alpha))
    print("Test sentences: " + str(test_sents1))
    print("Perplexity: " + str(compute_ppl_(model_sb, test_sents1)))

    print("\n\n")
    # Test the models with some example sentences
    test_sentences = [['the', 'king', 'is', 'dead'], ['the', 'queen', 'is', 'alive']]
    for sentence in test_sentences:
        print(f'Sentence: {sentence}\n')
        for model in [model_sb]:
            print(f'Model: {model.__class__.__name__}')
            for i in range(len(sentence) - n + 1):
                context = tuple(sentence[i:i + n - 1])
                word = sentence[i + n - 1]
                if word in vocab:
                    print(f'"{word}" given "{context}" has probability {model.score(word, context):.4f}')
                else:
                    print(f'"{word}" given "{context}" is an OOV word')
            print()

    ##########################################################################################################
    # 2. Own Stupid Backoff Implementation, Model Training and Testing
    ##########################################################################################################
    # Calling the MyStupidBackoff class
    my_model = MyStupidBackoff(macbeth_words, n=2, alpha=0.4)
    print("\n\n")
    # Test the models with some example sentences
    # Test the models with some example sentences
    test_sentences = [['the', 'king', 'is', 'dead'], ['the', 'queen', 'is', 'alive']]
    for sentence in test_sentences:
        print(f'Sentence: {sentence}\n')
        for model in [my_model]:
            print(f'Model: {model.__class__.__name__}')
            for i in range(len(sentence) - n + 1):
                context = tuple(sentence[i:i + n - 1])
                word = sentence[i + n - 1]
                if word in vocab:
                    print(f'"{word}" given "{context}" has probability {model.score(word, context):.4f}')
                else:
                    print(f'"{word}" given "{context}" is an OOV word')
            print()

    # Compute perplexity of the My Stupid Backoff model
    my_sb_perplexity = compute_ppl_my_sb(my_model, test_sentences)
    print(f"\nPerplexity of My Stupid Backoff model: {my_sb_perplexity:.4f}")

    ##########################################################################################################
    # Final Comparisons Between the NLTK's Stupid Backoff and My Stupid Backoff
    ##########################################################################################################
    print("\n\n")
    print("****Final Comparisons Between the NLTK's Stupid Backoff and My Stupid Backoff****")
    # Test the models with some example sentences
    test_sentences = [['the', 'king', 'is', 'dead'], ['the', 'queen', 'is', 'alive']]
    for sentence in test_sentences:
        print(f'Sentence: {sentence}\n')
        for model in [model_sb, my_model]:
            print(f'Model: {model.__class__.__name__}')
            for i in range(len(sentence) - n + 1):
                context = tuple(sentence[i:i + n - 1])
                word = sentence[i + n - 1]
                if word in vocab:
                    print(f'"{word}" given "{context}" has probability {model.score(word, context):.4f}')
                else:
                    print(f'"{word}" given "{context}" is an OOV word')
            print()

    # Compute perplexity of the My Stupid Backoff model
    print(f"Perplexity of NLTK's Stupid Backoff model: {compute_ppl_(model_sb, test_sents2):.4f}")
    print(f"Perplexity of My Stupid Backoff model: {my_sb_perplexity:.4f}")
