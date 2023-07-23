def statistics(chars, words, sents):
    """
    Compute descriptive statistics on the reference (.raw, words, etc.) sentences and tokens.
    :param chars: A list of characters
    :param words: A list of words
    :param sents: A list of sentences
    :return: A tuple of descriptive statistics
    """
    word_lens = [len(word) for word in words]
    sent_lens = [len(sent) for sent in sents]
    chars_len_in_sents = [len(''.join(sent)) for sent in sents]

    # total number of characters
    total_num_chars = len(chars)
    # total number of words (tokens: includes punctuation, etc.)
    total_num_words = len(words)
    # total number of sentences
    total_num_sents = len(sents)

    # minimum/maximum/average number of character per token
    min_char_per_token = min(word_lens)
    max_char_per_token = max(word_lens)
    avg_char_per_token = round(sum(word_lens) / len(words))

    # minimum/maximum/average number of words per sentence
    min_word_per_sent = min(sent_lens)
    max_word_per_sent = max(sent_lens)
    avg_word_per_sent = round(sum(sent_lens) / len(sents))

    # minimum/maximum/average number of sentences per document
    min_sent_per_doc = min(chars_len_in_sents)
    max_sent_per_doc = max(chars_len_in_sents)
    avg_sent_per_doc = round(sum(chars_len_in_sents) / len(chars_len_in_sents))

    return total_num_chars, total_num_words, total_num_sents, min_char_per_token, max_char_per_token, \
        avg_char_per_token, min_word_per_sent, max_word_per_sent, avg_word_per_sent, min_sent_per_doc, \
        max_sent_per_doc, avg_sent_per_doc
