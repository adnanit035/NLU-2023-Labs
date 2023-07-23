import os
import sys

from sklearn_crfsuite import CRF

from conll import evaluate
import pandas as pd
from spacy.tokenizer import Tokenizer
import es_core_news_sm
from nltk.corpus import conll2002

from functions import *


sys.path.insert(0, os.path.abspath('../src/'))


if __name__ == "__main__":
    nlp = es_core_news_sm.load()

    # nlp = spacy.load("es_core_news_sm")
    nlp.tokenizer = Tokenizer(nlp.vocab)  # to use white space tokenization (generally a bad idea for unknown data)

    # let's get only word and iob-tag
    trn_sents = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.train')]
    tst_sents = [[(text, iob) for text, pos, iob in sent] for sent in conll2002.iob_sents('esp.testa')]

    trn_feats = [sent2spacy_features(s, nlp) for s in trn_sents]
    trn_label = [sent2labels(s) for s in trn_sents]

    tst_feats = [sent2spacy_features(s, nlp) for s in tst_sents]
    ###############################################################################################################
    # 1. Baseline using the fetures in sent2spacy_features
    # 	- Train the model and print results on the test set
    ###############################################################################################################
    # train the model
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    try:
        crf.fit(trn_feats, trn_label)
    except AttributeError:
        pass

    # predict
    pred = crf.predict(tst_feats)

    # evaluate
    hyp = [[(tst_feats[i][j], t) for j, t in enumerate(tokens)] for i, tokens in enumerate(pred)]
    results = evaluate(tst_sents, hyp)

    pd_tbl = pd.DataFrame().from_dict(results, orient='index')
    print("\n\n********** Baseline Results **********")
    print(pd_tbl.head())

    ###############################################################################################################
    # 2. Add the "suffix" feature
    ###############################################################################################################
    # 2.1. Add the "suffix" feature to sent2spacy_features
    trn_feats = [sent2spacy_features_with_suffix(s, nlp) for s in trn_sents]
    trn_label = [sent2labels(s) for s in trn_sents]

    tst_feats = [sent2spacy_features_with_suffix(s, nlp) for s in tst_sents]
    # train the model
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )
    # %%
    try:
        crf.fit(trn_feats, trn_label)
    except AttributeError:
        pass

    pred = crf.predict(tst_feats)

    hyp = [[(tst_feats[i][j], t) for j, t in enumerate(tokens)] for i, tokens in enumerate(pred)]

    results = evaluate(tst_sents, hyp)

    pd_tbl = pd.DataFrame().from_dict(results, orient='index')

    print("\n\n********** Results with suffix **********")
    print(pd_tbl.head())

    ###############################################################################################################
    # 3. Add all the features used in the tutorial on CoNLL dataset
    ###############################################################################################################
    trn_feats = [sent2spacy_features_combined(s, nlp) for s in trn_sents]
    trn_label = [sent2labels(s) for s in trn_sents]

    tst_feats = [sent2spacy_features_combined(s, nlp) for s in tst_sents]
    trn_feats[0]

    # train the model
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    # workaround for scikit-learn 1.0
    try:
        crf.fit(trn_feats, trn_label)
    except AttributeError:
        pass

    pred = crf.predict(tst_feats)

    hyp = [[(tst_feats[i][j], t) for j, t in enumerate(tokens)] for i, tokens in enumerate(pred)]

    results = evaluate(tst_sents, hyp)

    pd_tbl = pd.DataFrame().from_dict(results, orient='index')

    print("\n\n********** Results with all features **********")
    print(pd_tbl.head())

    ###############################################################################################################
    # 4. Increase the feature window (number of previous and next token) to: [-1, +1]
    ###############################################################################################################
    trn_feats = [sent2spacy_features_combined_with_increased_window_one_one(s, nlp) for s in trn_sents]
    trn_label = [sent2labels(s) for s in trn_sents]

    tst_feats = [sent2spacy_features_combined_with_increased_window_one_one(s, nlp) for s in tst_sents]
    trn_feats[0]

    # train the model
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    # workaround for scikit-learn 1.0
    try:
        crf.fit(trn_feats, trn_label)
    except AttributeError:
        pass

    # predict
    pred = crf.predict(tst_feats)

    hyp = [[(tst_feats[i][j], t) for j, t in enumerate(tokens)] for i, tokens in enumerate(pred)]

    results = evaluate(tst_sents, hyp)

    pd_tbl = pd.DataFrame().from_dict(results, orient='index')
    pd_tbl.round(decimals=3)

    print("\n\n********** Results with increased window **********")
    print(pd_tbl.head())

    ###############################################################################################################
    # 5. Increase the feature window (number of previous and next token) to: [-2, +2]
    ###############################################################################################################
    trn_feats = [sent2spacy_features_combined_with_increased_window_two_two(s, nlp) for s in trn_sents]
    trn_label = [sent2labels(s) for s in trn_sents]

    tst_feats = [sent2spacy_features_combined_with_increased_window_two_two(s, nlp) for s in tst_sents]

    # train the model
    crf = CRF(
        algorithm='lbfgs',
        c1=0.1,
        c2=0.1,
        max_iterations=100,
        all_possible_transitions=True
    )

    try:
        crf.fit(trn_feats, trn_label)
    except AttributeError:
        pass

    # predict
    pred = crf.predict(tst_feats)

    hyp = [[(tst_feats[i][j], t) for j, t in enumerate(tokens)] for i, tokens in enumerate(pred)]

    results = evaluate(tst_sents, hyp)

    pd_tbl = pd.DataFrame().from_dict(results, orient='index')
    pd_tbl.round(decimals=3)

    print("\n\n********** Results with increased window **********")
    print(pd_tbl.head())
