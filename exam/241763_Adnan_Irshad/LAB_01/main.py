from functions import *
import spacy
import nltk
nltk.download('gutenberg')
nltk.download('punkt')


if __name__ == "__main__":
    # 1. Load another corpus from Gutenberg (e.g. milton-paradise.txt)
    whitman_chars = nltk.corpus.gutenberg.raw('whitman-leaves.txt')
    whitman_words = nltk.corpus.gutenberg.words('whitman-leaves.txt')
    whitman_sents = nltk.corpus.gutenberg.sents('whitman-leaves.txt')

    print("Compute descriptive statistics on the reference (.raw, .words, etc.) sentences and tokens.")
    # calling the statistics function
    total_num_chars_, total_num_words_, total_num_sents_, min_char_per_token_, max_char_per_token_, avg_char_per_token_, \
        min_word_per_sent_, max_word_per_sent_, avg_word_per_sent_, min_sent_per_doc_, max_sent_per_doc_, \
        avg_sent_per_doc_ = statistics(whitman_chars, whitman_words, whitman_sents)

    print('Number of characters:', total_num_chars_)
    print('Number of words:', total_num_words_)
    print('Number of sentences:', total_num_sents_)
    print()

    print('Minimum number of characters per token:', min_char_per_token_)
    print('Maximum number of characters per token:', max_char_per_token_)
    print('Average number of characters per token:', avg_char_per_token_)
    print()

    print('Minimum number of words per sentence:', min_word_per_sent_)
    print('Maximum number of words per sentence:', max_word_per_sent_)
    print('Average number of words per sentence:', avg_word_per_sent_)
    print()

    print('Minimum number of sentences per document:', min_sent_per_doc_)
    print('Maximum number of sentences per document:', max_sent_per_doc_)
    print('Average number of sentences per document:', avg_sent_per_doc_)

    ######################################################################################################
    # 1. Compute descriptive statistics in the automatically processed corpus both with spacy and nltk
    ######################################################################################################
    # a) Descriptive Statistics with NLTK
    # Convert war text into sentences and create document(s) using NLTK
    sentences = nltk.sent_tokenize(whitman_chars)
    documents = [' '.join(sentences[i:i + 10]) for i in range(0, len(sentences), 30)]

    # Initialize variables for statistics
    num_characters = 0
    num_tokens = 0
    num_sentences = 0
    num_docs = len(documents)
    char_per_token = []
    words_per_sentence = []
    sentences_per_doc = []

    # Process each document in the corpus
    for doc_text in documents:
        # Tokenize the text into sentences and words
        sentences = nltk.sent_tokenize(doc_text)
        words = nltk.word_tokenize(doc_text)

        # Calculate statistics for the current document
        num_characters += len(doc_text)
        num_tokens += len(words)
        num_sentences += len(sentences)

        char_per_token += [len(token) for token in words]
        words_per_sentence += [len(nltk.word_tokenize(sent)) for sent in sentences]
        sentences_per_doc.append(len(sentences))

    # Compute statistics
    avg_char_per_token = sum(char_per_token) / len(char_per_token)
    avg_words_per_sentence = sum(words_per_sentence) / len(words_per_sentence)
    avg_sentences_per_doc = sum(sentences_per_doc) / len(sentences_per_doc)

    print("\n\nCompute descriptive statistics in the automatically processed corpus both with spacy and nltk:\n")
    print(f"Total number of characters: {num_characters}")
    print(f"Total number of tokens: {num_tokens}")
    print(f"Total number of sentences: {num_sentences}")
    print()

    print(f"Minimum number of characters per token: {min(char_per_token)}")
    print(f"Maximum number of characters per token: {max(char_per_token)}")
    print(f"Average number of words per sentence: {avg_words_per_sentence:.2f}")
    print()

    print(f"Minimum number of sentences per document: {min(sentences_per_doc)}")
    print(f"Maximum number of sentences per document: {max(sentences_per_doc)}")
    print(f"Average number of sentences per document: {avg_sentences_per_doc:.2f}")

    # b) Descriptive Statistics with Spacy
    # Convert war text into sentences and create document(s) using Spacy
    nlp = spacy.load('en_core_web_sm')  # load the English language model

    doc = nlp(whitman_chars)
    documents = [sent.text for sent in doc.sents]  # create a list of sentences
    documents

    # Initialize variables for statistics
    num_characters = 0
    num_tokens = 0
    num_sentences = 0
    num_docs = len(documents)
    char_per_token = []
    words_per_sentence = []
    sentences_per_doc = []

    # Process each document in the corpus
    for doc_text in documents:
        doc = nlp(doc_text)

        # Calculate statistics for the current document
        num_characters += len(doc_text)
        num_tokens += len(doc)
        num_sentences += len(list(doc.sents))

        char_per_token += [len(token) for token in doc if not token.is_space]
        words_per_sentence += [len(sent) for sent in doc.sents]
        sentences_per_doc.append(len(list(doc.sents)))

    # Compute statistics
    avg_char_per_token = sum(char_per_token) / len(char_per_token)
    avg_words_per_sentence = sum(words_per_sentence) / len(words_per_sentence)
    avg_sentences_per_doc = sum(sentences_per_doc) / len(sentences_per_doc)

    # Print results
    print(f"Total number of characters: {num_characters}")
    print(f"Total number of tokens: {num_tokens}")
    print(f"Total number of sentences: {num_sentences}")

    print()
    print(f"Minimum number of characters per token: {min(char_per_token)}")
    print(f"Maximum number of characters per token: {max(char_per_token)}")
    print(f"Average number of characters per token: {avg_char_per_token:.2f}")

    print()
    print(f"Minimum number of words per sentence: {min(words_per_sentence)}")
    print(f"Maximum number of words per sentence: {max(words_per_sentence)}")
    print(f"Average number of words per sentence: {avg_words_per_sentence:.2f}")

    print()
    print(f"Minimum number of sentences per document: {min(sentences_per_doc)}")
    print(f"Maximum number of sentences per document: {max(sentences_per_doc)}")
    print(f"Average number of sentences per document: {avg_sentences_per_doc:.2f}")

    ######################################################################################################
    # 3. Compute lower-cased lexicons for all 3 (reference, spacy and nltk) versions of the corpus compare lexicon sizes
    ######################################################################################################
    # reference corpus
    whitman_words_lower = [w.lower() for w in whitman_words]
    text = " ".join(whitman_words_lower)

    reference_lower_lexicons = set(whitman_words_lower)

    # spacy corpus
    nlp = spacy.load('en_core_web_sm')
    whitman_words_spacy = nlp(text)
    spacy_lower_lexicons = set(whitman_words_spacy)

    # nltk corpus
    whitman_words_nltk = nltk.word_tokenize(text)
    nltk_lower_lexicons = set(whitman_words_nltk)

    print("\n3. Compute lowercased lexicons for all 3 versions (reference, spacy, nltk) of the corpus")
    print("Compare lexicon sizes:")
    # Computing lexicon Sizes
    # reference corpus size
    print('Reference corpus size:')
    print('Number of words:', len(reference_lower_lexicons))

    # spacy corpus size
    print('Spacy corpus size:')
    print('Number of words:', len(spacy_lower_lexicons))

    # nltk corpus size
    print('Nltk corpus size:')
    print('Number of words:', len(nltk_lower_lexicons))

    #############################################################################################
    # 4.Compute frequency distribution for all 3 (reference, spacy and nltk)
    #############################################################################################
    # Calculating frequency distribution
    whitman_words_lower_freq = nltk.FreqDist(whitman_words)
    # spacy corpus
    whitman_words_spacy_freq = nltk.FreqDist(whitman_words_spacy)
    # nltk corpus
    whitman_words_nltk_freq = nltk.FreqDist(whitman_words_nltk)

    print("\n4. Compute frequency distribution for all 3 (reference, spacy and nltk) versions of the corpus")
    print("Calulating frequency distribution for reference corpus")
    # let N is 10
    N = 10
    # getting top N frequencies reference corpus
    print('Reference corpus top 10 frequencies:')
    print(whitman_words_lower_freq.most_common(N))

    # spacy corpus
    print('Spacy corpus top 10 frequencies:')
    print(whitman_words_spacy_freq.most_common(N))

    # nltk corpus
    print('Nltk corpus top 10 frequencies:')
    print(whitman_words_nltk_freq.most_common(N))
