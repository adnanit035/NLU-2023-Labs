from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
import nltk
from nltk.util import ngrams
from nltk.corpus import senseval
from sklearn.feature_extraction import DictVectorizer
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from collections import Counter
from nltk.corpus import wordnet_ic
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from nltk.corpus import wordnet

semcor_ic = wordnet_ic.ic('ic-semcor.dat')


def extended_collocational_features(inst, ngram_range=3):
    p = inst.position
    features = {
        "w-2_word": 'NULL' if p < 2 else inst.context[p-2][0],
        "w-1_word": 'NULL' if p < 1 else inst.context[p-1][0],
        "w+1_word": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][0],
        "w+2_word": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][0],
        "w-2_tag": 'NULL' if p < 2 else inst.context[p-2][1],
        "w-1_tag": 'NULL' if p < 1 else inst.context[p-1][1],
        "w+1_tag": 'NULL' if len(inst.context) - 1 < p+1 else inst.context[p+1][1],
        "w+2_tag": 'NULL' if len(inst.context) - 1 < p+2 else inst.context[p+2][1],
    }

    # Adding ngrams
    start, end = max(0, p - ngram_range + 1), min(len(inst.context), p + ngram_range)
    ngram_list = list(ngrams([t[0] for t in inst.context[start:end]], ngram_range))
    for i, ngram in enumerate(ngram_list):
        features[f"ngram_{i}"] = " ".join(ngram)

    return features


def preprocess(sentence):
    mapping = {"NOUN": wordnet.NOUN, "VERB": wordnet.VERB, "ADJ": wordnet.ADJ, "ADV": wordnet.ADV}

    sw_list = stopwords.words('english')

    lemmatizer = WordNetLemmatizer()

    tokens = nltk.word_tokenize(sentence) if isinstance(sentence, str) else sentence

    tagged = nltk.pos_tag(tokens, tagset="universal")

    # lowercase
    tagged = [(w.lower(), p) for w, p in tagged]
    # optional: remove all words that are not NOUN, VERB, ADJ, or ADV (i.e. no sense in WordNet)
    tagged = [(w, p) for w, p in tagged if p in mapping]
    # re-map tags to WordNet (return orignal if not in-mapping, if above is not used)
    tagged = [(w, mapping.get(p, p)) for w, p in tagged]
    # remove stopwords
    tagged = [(w, p) for w, p in tagged if w not in sw_list]
    # lemmatize
    tagged = [(w, lemmatizer.lemmatize(w, pos=p), p) for w, p in tagged]
    # unique the list
    tagged = list(set(tagged))

    return tagged


def get_sense_definitions(context):
    # input is text or list of strings
    lemma_tags = preprocess(context)

    # get all possible senses for each word in context
    senses_ = []
    for lemma_, _, tag in lemma_tags:
        senses_.append((lemma_, wordnet.synsets(lemma_, pos=tag)))

    # get all possible sense definitions for each word in context
    sense_defs = []
    for w_, senses_list_ in senses_:
        if len(senses_list_) > 0:
            defs_list_ = []
            for sense_ in senses_list_:
                defs_ = sense_.definition()

                tags_ = preprocess(defs_)

                # get tokens from tags
                tokens_ = [w for w, _, _ in tags_]

                defs_list_.append((sense_, tokens_))

            sense_defs.append((w_, defs_list_))

    return sense_defs


def get_top_sense(words, sense_list):
    val, sense_ = max((len(set(words).intersection(set(defn))), ss) for ss, defn in sense_list)
    return val, sense_


def original_lesk(context_sentence, ambiguous_word, pos=None, synsets=None, majority=False):

    context_senses = get_sense_definitions(set(context_sentence) - set([ambiguous_word]))
    if synsets is None:
        try:
            synsets = get_sense_definitions(ambiguous_word)[0][1]
        except:
            return None

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None

    scores = []
    # print(synsets)
    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense(sense[1], synsets))

    if len(scores) == 0:
        return synsets[0][0]

    if majority:
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            # Almost random selection
            best_sense = Counter([x[1] for x in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)
    return best_sense


# 3.2 Graph-based Lesk (Lesk Similarity)
def get_top_sense_sim(context_sense, sense_list, similarity):
    # get top sense from the list of sense-definition tuples
    # assumes that words and definitions are preprocessed identically
    scores = []
    for sense in sense_list:
        ss = sense[0]
        if similarity == "path":
            try:
                # Append path similarity between ambiguous word and senses from the context
                # calculate path similarity between two senses
                path_similarity_score = context_sense.path_similarity(ss)
                scores.append((path_similarity_score, ss))
            except:
                scores.append((0, ss))
        elif similarity == "lch":
            try:
                # Append LCH similarity between ambiguous word and senses from the context
                lch_similarity_score = context_sense.lch_similarity(ss)
                scores.append((lch_similarity_score, ss))
            except:
                scores.append((0, ss))
        elif similarity == "wup":
            try:
                # Append WUP similarity between ambiguous word and senses from the context
                wup_similarity_score = context_sense.wup_similarity(ss)
                scores.append((wup_similarity_score, ss))
            except:
                scores.append((0, ss))

        elif similarity == "resnik":
            try:
                # Append Resnik similarity  between ambiguous word and senses from the context
                # Don't forget semcor_ic
                resnik_similarity_score = context_sense.res_similarity(ss, semcor_ic)
                scores.append((resnik_similarity_score, ss))
            except:
                scores.append((0, ss))
        elif similarity == "lin":
            try:
                # Append lin similarity between ambiguous word and senses from the context
                # Don't forget semcor_ic
                lin_similarity_score = context_sense.lin_similarity(ss, semcor_ic)
                scores.append((lin_similarity_score, ss))
            except:
                scores.append((0, ss))

        elif similarity == "jiang":
            try:
                # Append Jiang similarity between ambiguous word and senses from the context
                # Don't forget semcor_ic
                ji_similarity_score = context_sense.ji_similarity(ss, semcor_ic)
                scores.append((ji_similarity_score, ss))
            except:
                scores.append((0, ss))
        else:
            print("Similarity metric not found")
            return None

    val, sense_ = max(scores)
    return val, sense_


def lesk_similarity(context_sentence, ambiguous_word, similarity="resnik", pos=None, synsets=None, majority=True):
    context_senses = get_sense_definitions(set(context_sentence) - set([ambiguous_word]))

    if synsets is None:
        try:
            synsets = get_sense_definitions(ambiguous_word)[0][1]
        except:
            # If no synsets are found, return None
            return None

    if pos:
        synsets = [ss for ss in synsets if str(ss[0].pos()) == pos]

    if not synsets:
        return None

    scores = []

    for senses in context_senses:
        for sense in senses[1]:
            scores.append(get_top_sense_sim(sense[0], synsets, similarity))

    if len(scores) == 0:
        return synsets[0][0]

    # Majority voting as before
    if majority:
        # We remove 0 scores, senses without overlapping
        filtered_scores = [x[1] for x in scores if x[0] != 0]
        if len(filtered_scores) > 0:
            # Select the most common syn.
            best_sense = Counter(filtered_scores).most_common(1)[0][0]
        else:
            # Almost random selection
            best_sense = Counter([score[1] for score in scores]).most_common(1)[0][0]
    else:
        _, best_sense = max(scores)

    return best_sense
