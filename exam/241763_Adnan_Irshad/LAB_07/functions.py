def sent2labels(sent):
    return [label for token, label in sent]


def sent2tokens(sent):
    return [token for token, label in sent]


def sent2spacy_features(sent, nlp):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for token in spacy_sent:
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'pos': token.pos_,
            'lemma': token.lemma_
        }
        feats.append(token_feats)

    return feats


def sent2spacy_features_with_suffix(sent, nlp):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for token in spacy_sent:
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'pos': token.pos_,
            'lemma': token.lemma_,
            'suffix': token.lower_[-3:]
        }
        feats.append(token_feats)

    return feats


def sent2spacy_features_combined(sent, nlp):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for token in spacy_sent:
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'word[-3:]': token.lower_[-3:],
            'word[-2:]': token.lower_[-2:],
            'word.isupper()': token.is_upper,
            'word.istitle()': token.is_title,
            'word.isdigit()': token.is_digit,
            'postag': token.pos_,
            'postag[:2]': token.pos_[:2],
            'lemma': token.lemma_,
            'suffix': token.lower_[-1:],
            'suffix2': token.lower_[-2:],
            'suffix3': token.lower_[-3:]
        }
        feats.append(token_feats)

    return feats


def sent2spacy_features_combined_with_increased_window_one_one(sent, nlp):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for i, token in enumerate(spacy_sent):
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'word[-3:]': token.lower_[-3:],
            'word[-2:]': token.lower_[-2:],
            'word.isupper()': token.is_upper,
            'word.istitle()': token.is_title,
            'word.isdigit()': token.is_digit,
            'postag': token.pos_,
            'postag[:2]': token.pos_[:2],
            'lemma': token.lemma_,
            'suffix': token.lower_[-1:],
            'suffix2': token.lower_[-2:],
            'suffix3': token.lower_[-3:]
        }
        if i > 0:
            token1 = spacy_sent[i - 1]
            token_feats.update({
                '-1:word.lower()': token1.lower_,
                '-1:word.istitle()': token1.is_title,
                '-1:word.isupper()': token1.is_upper,
                '-1:postag': token1.pos_,
                '-1:postag[:2]': token1.pos_[:2],
                '-1:lemma': token1.lemma_,
                '-1:suffix': token1.lower_[-1:],
                '-1:suffix2': token1.lower_[-2:],
                '-1:suffix3': token1.lower_[-3:]
            })
        else:
            token_feats['BOS'] = True

        if i < len(spacy_sent) - 1:
            token1 = spacy_sent[i + 1]
            token_feats.update({
                '+1:word.lower()': token1.lower_,
                '+1:word.istitle()': token1.is_title,
                '+1:word.isupper()': token1.is_upper,
                '+1:postag': token1.pos_,
                '+1:postag[:2]': token1.pos_[:2],
                '+1:lemma': token1.lemma_,
                '+1:suffix': token1.lower_[-1:],
                '+1:suffix2': token1.lower_[-2:],
                '+1:suffix3': token1.lower_[-3:]
            })
        else:
            token_feats['EOS'] = True

        feats.append(token_feats)

    return feats


# define the features in sent2spacy_features
def sent2spacy_features_combined_with_increased_window_two_two(sent, nlp):
    spacy_sent = nlp(" ".join(sent2tokens(sent)))
    feats = []
    for i, token in enumerate(spacy_sent):
        token_feats = {
            'bias': 1.0,
            'word.lower()': token.lower_,
            'word[-3:]': token.lower_[-3:],
            'word[-2:]': token.lower_[-2:],
            'word.isupper()': token.is_upper,
            'word.istitle()': token.is_title,
            'word.isdigit()': token.is_digit,
            'postag': token.pos_,
            'postag[:2]': token.pos_[:2],
            'lemma': token.lemma_,
            'suffix': token.lower_[-1:],
            'suffix2': token.lower_[-2:],
            'suffix3': token.lower_[-3:]
        }
        if i > 0:
            token1 = spacy_sent[i-1]
            token_feats.update({
                '-1:word.lower()': token1.lower_,
                '-1:word.istitle()': token1.is_title,
                '-1:word.isupper()': token1.is_upper,
                '-1:postag': token1.pos_,
                '-1:postag[:2]': token1.pos_[:2],
                '-1:lemma': token1.lemma_,
                '-1:suffix': token1.lower_[-1:],
                '-1:suffix2': token1.lower_[-2:],
                '-1:suffix3': token1.lower_[-3:]
            })
        else:
            token_feats['BOS'] = True

        if i < len(spacy_sent)-1:
            token1 = spacy_sent[i+1]
            token_feats.update({
                '+1:word.lower()': token1.lower_,
                '+1:word.istitle()': token1.is_title,
                '+1:word.isupper()': token1.is_upper,
                '+1:postag': token1.pos_,
                '+1:postag[:2]': token1.pos_[:2],
                '+1:lemma': token1.lemma_,
                '+1:suffix': token1.lower_[-1:],
                '+1:suffix2': token1.lower_[-2:],
                '+1:suffix3': token1.lower_[-3:]
            })
        else:
            token_feats['EOS'] = True

        if i > 1:
            token1 = spacy_sent[i-2]
            token_feats.update({
                '-2:word.lower()': token1.lower_,
                '-2:word.istitle()': token1.is_title,
                '-2:word.isupper()': token1.is_upper,
                '-2:postag': token1.pos_,
                '-2:postag[:2]': token1.pos_[:2],
                '-2:lemma': token1.lemma_,
                '-2:suffix': token1.lower_[-1:],
                '-2:suffix2': token1.lower_[-2:],
                '-2:suffix3': token1.lower_[-3:]
            })
        else:
            token_feats['BOS'] = True

        if i < len(spacy_sent)-2:
            token1 = spacy_sent[i+2]
            token_feats.update({
                '+2:word.lower()': token1.lower_,
                '+2:word.istitle()': token1.is_title,
                '+2:word.isupper()': token1.is_upper,
                '+2:postag': token1.pos_,
                '+2:postag[:2]': token1.pos_[:2],
                '+2:lemma': token1.lemma_,
                '+2:suffix': token1.lower_[-1:],
                '+2:suffix2': token1.lower_[-2:],
                '+2:suffix3': token1.lower_[-3:]
            })
        else:
            token_feats['EOS'] = True

        feats.append(token_feats)

    return feats
