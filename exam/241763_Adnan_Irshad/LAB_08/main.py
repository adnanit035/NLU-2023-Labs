from functions import *

nltk.download('senseval')
# getting pre-computed ic of the semcor corpus (large sense tagged corpus)
nltk.download('wordnet_ic')
semcor_ic = wordnet_ic.ic('ic-semcor.dat')


if __name__ == "__main__":
    # Load the data
    inst = senseval.instances('interest.pos')[0]
    data = [" ".join([t[0] for t in inst.context]) for inst in senseval.instances('interest.pos')]
    lbls = [inst.senses[0] for inst in senseval.instances('interest.pos')]

    #############################################################################################################
    # 1 - Extend collocational features with
    # 	- POS-tags
    # 	- Ngrams within window
    #############################################################################################################
    # 1.1 Collocational Features
    # 	- Context Words
    # 	- POS-tags of these words
    # 	- word ngrams in window +/-3 are common

    print("\n\n ****** 1. Extended Collocational Features ****** \n\n")
    data_col_extended = [extended_collocational_features(inst) for inst in senseval.instances('interest.pos')]
    print(data_col_extended[0])

    vectorizer = CountVectorizer()
    classifier = MultinomialNB()
    lblencoder = LabelEncoder()

    # stratified split
    stratified_split = StratifiedKFold(n_splits=5, shuffle=True)
    vectors = vectorizer.fit_transform(data)

    # encoding labels for multi-calss
    lblencoder.fit(lbls)
    labels = lblencoder.transform(lbls)

    # evaluate the performance of the extended collocational features
    dvectorizer = DictVectorizer(sparse=False)
    dvectors = dvectorizer.fit_transform(data_col_extended)

    scores = cross_validate(classifier, dvectors, labels, cv=stratified_split, scoring=['f1_micro'])
    print(sum(scores['test_f1_micro']) / len(scores['test_f1_micro']))

    #############################################################################################################
    # 2: Bag of Words Classification
    #############################################################################################################
    print("\n\n ****** 2. Bag of Words Classification ****** \n\n")
    vectorizer = CountVectorizer()
    classifier = MultinomialNB()
    lblencoder = LabelEncoder()

    stratified_split = StratifiedKFold(n_splits=5, shuffle=True)

    vectors = vectorizer.fit_transform(data)

    # encoding labels for multi-calss
    lblencoder.fit(lbls)
    labels = lblencoder.transform(lbls)

    scores = cross_validate(classifier, vectors, labels, cv=stratified_split, scoring=['f1_micro'])

    print(sum(scores['test_f1_micro']) / len(scores['test_f1_micro']))
    #############################################################################################################
    # 2. Concatenating Feature Vectors (BOW + New Collocation)
    #############################################################################################################
    uvectors = np.concatenate((vectors.toarray(), dvectors), axis=1)

    # cross-validating classifier the usual way
    scores = cross_validate(classifier, uvectors, labels, cv=stratified_split, scoring=['f1_micro'])
    print("\n\n ****** 3. Concatenating Feature Vectors (BOW + New Collocation) ****** \n\n")
    print(sum(scores['test_f1_micro']) / len(scores['test_f1_micro']))

    #############################################################################################################
    # 3. Evaluate Lesk Original and Graph-based (Lesk Similarity or Pedersen) metrics on the same test split and compare
    #############################################################################################################
    print("\n\n ****** 4. Lesk Original and Graph-based ****** \n\n")

    # # Evaluation of Original Lesk and Graph-based (Lesk Similarity) metrics on the same test split and compare
    # # k-fold cross validation
    data = senseval.instances('interest.pos')
    mapping = {
        'interest.n.01': 'interest_1',
        'interest.n.03': 'interest_2',
        'pastime.n.01': 'interest_3',
        'sake.n.01': 'interest_4',
        'interest.n.05': 'interest_5',
        'interest.n.04': 'interest_6',

    }

    # since WordNet defines more senses, let's restrict predictions
    synsets = []
    for ss in wordnet.synsets('interest', pos='n'):
        if ss.name() in mapping.values():
            defn = ss.definition()
            tags = preprocess(defn)
            toks = [l for w, l, p in tags]
            synsets.append((ss, toks))

    kf = KFold(n_splits=10, shuffle=True)
    kf.get_n_splits(data)

    original_lesk_accuracy = []
    original_lesk_precision = []
    original_lesk_recall = []
    original_lesk_f1 = []

    lesk_similarity_accuracy = []
    lesk_similarity_precision = []
    lesk_similarity_recall = []
    lesk_similarity_f1 = []

    for train_index, test_index in kf.split(data):
        train_data = [data[i] for i in train_index]
        test_data = [data[i] for i in test_index]

        original_lesk_predictions = []
        lesk_similarity_predictions = []

        original_lesk_true = []
        lesk_similarity_true = []

        for instance in test_data:
            sentence_ = " ".join([t[0] for t in instance.context])
            word_ = str(instance.word).split('-')[0]
            pred = original_lesk(sentence_.split(), word_)
            if pred is not None:
                pred = pred.name()
                if pred in mapping:
                    pred = mapping[pred]
                original_lesk_predictions.append(pred)

            pred = lesk_similarity(sentence_.split(), word_, similarity="resnik", pos='n', majority=True)
            if pred is not None:
                pred = pred.name()
                if pred in mapping:
                    pred = mapping[pred]

                lesk_similarity_predictions.append(pred)

            original_lesk_true.append(instance.senses[0])
            lesk_similarity_true.append(instance.senses[0])

        try:
            original_lesk_accuracy.append(
                accuracy_score(original_lesk_true, original_lesk_predictions))
        except:
            original_lesk_accuracy.append(0)
        try:
            original_lesk_precision.append(
                precision_score(original_lesk_true, original_lesk_predictions, average='macro', zero_division=0))
        except:
            original_lesk_precision.append(0)
        try:
            original_lesk_recall.append(
                recall_score(original_lesk_true, original_lesk_predictions, average='macro', zero_division=0))
        except:
            original_lesk_recall.append(0)
        try:
            original_lesk_f1.append(
                f1_score(original_lesk_true, original_lesk_predictions, average='macro', zero_division=0))
        except:
            original_lesk_f1.append(0)

        try:
            lesk_similarity_accuracy.append(
                accuracy_score(lesk_similarity_true, lesk_similarity_predictions))
        except:
            lesk_similarity_accuracy.append(0)
        try:
            lesk_similarity_precision.append(
                precision_score(lesk_similarity_true, lesk_similarity_predictions, average='macro', zero_division=0))
        except:
            lesk_similarity_precision.append(0)
        try:
            lesk_similarity_recall.append(
                recall_score(lesk_similarity_true, lesk_similarity_predictions, average='macro', zero_division=0))
        except:
            lesk_similarity_recall.append(0)
        try:
            lesk_similarity_f1.append(
                f1_score(lesk_similarity_true, lesk_similarity_predictions, average='macro', zero_division=0))
        except:
            lesk_similarity_f1.append(0)

    print("Original Lesk Algorithm Results:")
    print("\t ==> Accuracy: ", np.mean(original_lesk_accuracy))
    print("\t ==> Precision: ", np.mean(original_lesk_precision))
    print("\t ==> Recall: ", np.mean(original_lesk_recall))
    print("\t ==> F1: ", np.mean(original_lesk_f1))

    print("\nLesk Similarity Algorithm Results:")
    print("\t ==> Accuracy: ", np.mean(lesk_similarity_accuracy))
    print("\t ==> Precision: ", np.mean(lesk_similarity_precision))
    print("\t ==> Recall: ", np.mean(lesk_similarity_recall))
    print("\t ==> F1: ", np.mean(lesk_similarity_f1))
