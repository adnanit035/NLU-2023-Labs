import warnings

import en_core_web_sm
import nltk
from nltk import NgramTagger
from nltk.corpus import treebank
from nltk.metrics import accuracy
from spacy.tokenizer import Tokenizer

warnings.filterwarnings("ignore", category=DeprecationWarning)

nltk.download('treebank')
nltk.download('universal_tagset')

if __name__ == "__main__":
    # Load the spacy model
    nlp = en_core_web_sm.load()

    # We overwrite the spacy tokenizer with a custom one, that split by whitespace only
    nlp.tokenizer = Tokenizer(nlp.vocab)  # Tokenize by whitespace

    # Split the treebank dataset into train and test sets
    dataset = treebank.tagged_sents(tagset='universal')
    trn_data = dataset[:3000]
    tst_data = dataset[3000:]

    ##############################################################################################################
    # 1. Train & Evaluate NGramTagger
    ##############################################################################################################
    # 1.1. NGramTagger with n=1,2,3 and cutt-off 3
    ngram_tagger = NgramTagger(1, trn_data)  # UnigramTagger
    accuracy_unigram = ngram_tagger.evaluate(tst_data)
    ngram_tagger_cutoff = NgramTagger(1, trn_data, cutoff=3)  # UnigramTagger with cut-off
    accuracy_unigram_cutoff = ngram_tagger_cutoff.evaluate(tst_data)

    ngram_tagger = NgramTagger(2, trn_data)  # BigramTagger
    accuracy_bigram = ngram_tagger.evaluate(tst_data)
    ngram_tagger = NgramTagger(2, trn_data, cutoff=3)  # BigramTagger with cut-off
    accuracy_bigram_cutoff = ngram_tagger.evaluate(tst_data)

    ngram_tagger = NgramTagger(3, trn_data)  # TrigramTagger
    accuracy_trigram = ngram_tagger.evaluate(tst_data)
    ngram_tagger = NgramTagger(3, trn_data, cutoff=3)  # TrigramTagger with cut-off
    accuracy_trigram_cutoff = ngram_tagger.evaluate(tst_data)

    print("\n****1. Train & Evaluate NGramTagger****")
    print("Unigram Tagger Accuracy: ", accuracy_unigram)
    print("Unigram Tagger Accuracy with cut-off: ", accuracy_unigram_cutoff)

    print("\nBigram Tagger Accuracy: ", accuracy_bigram)
    print("Bigram Tagger Accuracy with cut-off: ", accuracy_bigram_cutoff)

    print("\nTrigram Tagger Accuracy: ", accuracy_trigram)
    print("Trigram Tagger Accuracy with cut-off: ", accuracy_trigram_cutoff)

    ##############################################################################################################
    # 2. Evaluate spacy POS-tags on the same test set
    ##############################################################################################################
    print("\n\n****2. Evaluate spacy POS-tags on the same test set****")
    # spacy to nltk mapping dictionary
    mapping_spacy_to_NLTK = {
        "ADJ": "ADJ",
        "ADP": "ADP",
        "ADV": "ADV",
        "AUX": "VERB",
        "CCONJ": "CONJ",
        "DET": "DET",
        "INTJ": "X",
        "NOUN": "NOUN",
        "NUM": "NUM",
        "PART": "PRT",
        "PRON": "PRON",
        "PROPN": "NOUN",
        "PUNCT": ".",
        "SCONJ": "CONJ",
        "SYM": "X",
        "VERB": "VERB",
        "X": "X"
    }

    # convert output to the required format: flatten into a list
    spacy_tags = []
    nltk_tags = []
    for sent in tst_data:
        tokens = [token[0] for token in sent]
        spacy_doc = nlp(" ".join(tokens))
        spacy_tags.extend([token.pos_ for token in spacy_doc])
        nltk_tags.extend([mapping_spacy_to_NLTK[token.pos_] for token in spacy_doc])

    print("Spacy Tags: ", spacy_tags[:10])
    print("NLTK Tags: ", nltk_tags[:10])

    # Evaluate using `accuracy` from `nltk.metrics`
    print("\n\n****Evaluate using `accuracy` from `nltk.metrics`****")
    # evaluate using accuracy from nltk.metrics
    spacy_accuracy = accuracy(nltk_tags, spacy_tags)
    # Print the accuracy results for both NLTK and Spacy taggers
    print("NLTK: Accuracy")
    print("Unigram Tagger: ", accuracy_unigram)
    print("Bigram Tagger: ", accuracy_bigram)
    print("Trigram Tagger: ", accuracy_trigram)

    print("\nSpacy: Accuracy")
    print("Spacy Tagger: ", spacy_accuracy)
