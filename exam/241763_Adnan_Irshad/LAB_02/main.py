from collections import Counter
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.datasets import fetch_20newsgroups  # 20 newsgroups dataset
from sklearn.svm import LinearSVC  # Linear SVM model (SVC)
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_validate


if __name__ == "__main__":
    # Load the dataset
    data = fetch_20newsgroups(subset='all', shuffle=True)

    print("\n Descriptive Analysis of the Dataset:")
    # no. of samples
    print("Number of samples: ", len(data.data))
    # no. of classes
    print("Number of classes: ", len(data.target_names))
    # class names
    print("Class names: ", data.target_names)
    # sample per class distribution
    print("Samples per class: {}".format(dict(Counter(list(data.target)))))
    # print the first sample features
    print("First sample features: {}".format(data.data[0]))
    # print the labels
    print("Labels: {}".format(data.target))
    # print the first sample label
    print("First sample label: {}".format(data.target[0]))
    # print the first sample class name
    print("First sample class name: {}".format(data.target_names[data.target[0]]))

    # 5-fold cross-validation
    stratified_k_fold = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # split the dataset
    stratified_split = stratified_k_fold.split(data.data, data.target)

    split_train = []
    split_test = []

    # get the splits
    for train_index, test_index in stratified_k_fold.split(data.data, data.target):
        split_train.append(train_index)
        split_test.append(test_index)

    print("Number of splits: ", len(split_train))
    print("Number of Train samples in the first split: ", len(split_train[0]))
    print("Number of Test samples in the first split: ", len(split_test[0]))

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.hist(data.target[split_train[0]])
    plt.title("Train set")

    plt.subplot(1, 2, 2)
    plt.hist(data.target[split_test[0]])
    plt.title("Test set")

    plt.show()

    #####################################################################################################
    # 1. Using Newsgroup dataset from scikit-learn train and evaluate Linear SVM (LinearSVC) model
    #####################################################################################################
    print("\nTask-1: Using Newsgroup dataset from scikit-learn train and evaluate Linear SVM (LinearSVC) model")
    # Count Vectorization
    vectorizer = CountVectorizer()

    # use fit_transform to 'learn' the features and vectorize the data
    vectors = vectorizer.fit_transform(data.data)

    # Linear SVM model
    model = LinearSVC()

    # train and evaluate the model
    for i in range(len(split_train)):
        # train the model
        model.fit(vectors[split_train[i]], data.target[split_train[i]])

        # evaluate the model
        y_pred = model.predict(vectors[split_test[i]])

        # print the classification report
        print('\033[1m', 'Split : ', i, '\033[0m')
        print(classification_report(data.target[split_test[i]], y_pred))
        print()

    # Linear SVM model
    model = LinearSVC(C=0.0001)

    # train and evaluate the model
    for i in range(len(split_train)):
        # train the model
        model.fit(vectors[split_train[i]], data.target[split_train[i]])

        # evaluate the model
        y_pred = model.predict(vectors[split_test[i]])

        # print the classification report
        print('\033[1m', 'Split : ', i, '\033[0m')
        print(classification_report(data.target[split_test[i]], y_pred))
        print()

    # 1.5 Cross-Validation Evaluation


    # Linear SVM model
    model = LinearSVC(C=0.0001)

    # cross-validation evaluation
    for metric_score in ['f1_macro', 'f1_micro', 'f1_weighted']:
        scores = cross_validate(model, vectors, data.target, cv=stratified_k_fold, scoring=[metric_score])
        print('\033[1m', metric_score.upper(), '\033[0m', ':',
              round(sum(scores['test_' + metric_score]) / len(scores['test_' + metric_score]), 2))

    ################################################################################################################
    # 2 Experiment with different vectorization methods and parameters.
    ################################################################################################################
    # 2.1 binary of Count Vectorization (CountVect)
    # Count Vectorization
    print("\n\n\nTask-2: Experiment with different vectorization methods and parameters.")
    print("2.1 binary of Count Vectorization (CountVect)")
    vectorizer = CountVectorizer(binary=True)

    # use fit_transform to 'learn' the features and vectorize the data
    vectors = vectorizer.fit_transform(data.data)

    # Linear SVM model
    model = LinearSVC(C=0.0001)

    # cross-validation evaluation
    for metric_score in ['f1_macro', 'f1_micro', 'f1_weighted']:
        scores = cross_validate(model, vectors, data.target, cv=stratified_k_fold, scoring=[metric_score])
        print('\033[1m', metric_score.upper(), '\033[0m', ':',
              round(sum(scores['test_' + metric_score]) / len(scores['test_' + metric_score]), 2))

    # 2.2 TF-IDF Transformation (TF-IDF)
    print("\n2.2 TF-IDF Transformation (TF-IDF)")
    transformer = TfidfTransformer()

    # use fit_transform to 'learn' the features and vectorize the data
    vectors = transformer.fit_transform(vectors)

    # Linear SVM model
    model = LinearSVC(C=0.0001)

    # cross-validation evaluation
    for metric_score in ['f1_macro', 'f1_micro', 'f1_weighted']:
        scores = cross_validate(model, vectors, data.target, cv=stratified_k_fold, scoring=[metric_score])
        print('\033[1m', metric_score.upper(), '\033[0m', ':',
              round(sum(scores['test_' + metric_score]) / len(scores['test_' + metric_score]), 2))

    # 2.3 TF-IDF Vectorization (TFIDF)
    print("\n2.3 TF-IDF Vectorization (TFIDF)")
    vectorizer = TfidfVectorizer()

    # use fit_transform to 'learn' the features and vectorize the data
    vectors = vectorizer.fit_transform(data.data)

    # Linear SVM model
    model = LinearSVC(C=0.0001)

    # cross-validation evaluation
    for metric_score in ['f1_macro', 'f1_micro', 'f1_weighted']:
        scores = cross_validate(model, vectors, data.target, cv=stratified_k_fold, scoring=[metric_score])
        print('\033[1m', metric_score.upper(), '\033[0m', ':',
              round(sum(scores['test_' + metric_score]) / len(scores['test_' + metric_score]), 2))

    ## 2.4 min and max cut-offs (CutOff)
    print("\n2.4 min and max cut-offs (CutOff)")
    ### 2.4.1 min and max cut-offs (CutOff) with Count Vectorization (CountVect)
    # Count Vectorization
    vectorizer = CountVectorizer(min_df=0.01, max_df=0.99)

    # use fit_transform to 'learn' the features and vectorize the data
    vectors = vectorizer.fit_transform(data.data)
    # Linear SVM model
    model = LinearSVC(C=0.0001)

    # cross-validation evaluation
    for metric_score in ['f1_macro', 'f1_micro', 'f1_weighted']:
        scores = cross_validate(model, vectors, data.target, cv=stratified_k_fold, scoring=[metric_score])
        print('\033[1m', metric_score.upper(), '\033[0m', ':',
              round(sum(scores['test_' + metric_score]) / len(scores['test_' + metric_score]), 2))

    ### 2.4.2 min and max cut-offs (CutOff) with TF-IDF Transformation (TF-IDF)
    print("\n2.4.2 min and max cut-offs (CutOff) with TF-IDF Transformation (TF-IDF)")
    # TF-IDF Transformation with min and max cut-offs
    transformer = TfidfTransformer()

    # use fit_transform to 'learn' the features and vectorize the data
    vectors = transformer.fit_transform(vectors)
    # Linear SVM model
    model = LinearSVC(C=0.0001)

    # cross-validation evaluation
    for metric_score in ['f1_macro', 'f1_micro', 'f1_weighted']:
        scores = cross_validate(model, vectors, data.target, cv=stratified_k_fold, scoring=[metric_score])
        print('\033[1m', metric_score.upper(), '\033[0m', ':',
              round(sum(scores['test_' + metric_score]) / len(scores['test_' + metric_score]), 2))

    ### 2.4.3 min and max cut-offs (CutOff) with TF-IDF Vectorization (TFIDF)
    print("\n2.4.3 min and max cut-offs (CutOff) with TF-IDF Vectorization (TFIDF)")
    # TF-IDF Vectorization with min and max cut-offs
    vectorizer = TfidfVectorizer(min_df=0.01, max_df=0.99)

    # use fit_transform to 'learn' the features and vectorize the data
    vectors = vectorizer.fit_transform(data.data)
    # Linear SVM model
    model = LinearSVC(C=0.0001)

    # cross-validation evaluation
    for metric_score in ['f1_macro', 'f1_micro', 'f1_weighted']:
        scores = cross_validate(model, vectors, data.target, cv=stratified_k_fold, scoring=[metric_score])
        print('\033[1m', metric_score.upper(), '\033[0m', ':',
              round(sum(scores['test_' + metric_score]) / len(scores['test_' + metric_score]), 2))

    ## 2.5 wihtout stop-words (WithoutStopWords)
    ### 2.5.1 wihtout stop-words (WithoutStopWords) with Count Vectorization (CountVect)
    print("\n2.5.1 wihtout stop-words (WithoutStopWords) with Count Vectorization (CountVect)")
    # Count Vectorization without stop-words
    vectorizer = CountVectorizer(stop_words='english')

    # use fit_transform to 'learn' the features and vectorize the data
    vectors = vectorizer.fit_transform(data.data)
    # Linear SVM model
    model = LinearSVC(C=0.0001)

    # cross-validation evaluation
    for metric_score in ['f1_macro', 'f1_micro', 'f1_weighted']:
        scores = cross_validate(model, vectors, data.target, cv=stratified_k_fold, scoring=[metric_score])
        print('\033[1m', metric_score.upper(), '\033[0m', ':',
              round(sum(scores['test_' + metric_score]) / len(scores['test_' + metric_score]), 2))

    ### 2.5.2 wihtout stop-words (WithoutStopWords) with TF-IDF Transformation (TF-IDF)
    print("\n2.5.2 wihtout stop-words (WithoutStopWords) with TF-IDF Transformation (TF-IDF)")
    # TF-IDF Transformation without stop-words
    transformer = TfidfTransformer()

    # use fit_transform to 'learn' the features and vectorize the data
    vectors = transformer.fit_transform(vectors)

    # Linear SVM model
    model = LinearSVC(C=0.0001)

    # cross-validation evaluation
    for metric_score in ['f1_macro', 'f1_micro', 'f1_weighted']:
        scores = cross_validate(model, vectors, data.target, cv=stratified_k_fold, scoring=[metric_score])
        print('\033[1m', metric_score.upper(), '\033[0m', ':',
              round(sum(scores['test_' + metric_score]) / len(scores['test_' + metric_score]), 2))
    ### 2.5.3 wihtout stop-words (WithoutStopWords) with TF-IDF Vectorization (TFIDF)
    print("\n2.5.3 wihtout stop-words (WithoutStopWords) with TF-IDF Vectorization (TFIDF)")
    # TF-IDF Vectorization without stop-words
    vectorizer = TfidfVectorizer(stop_words='english')

    # use fit_transform to 'learn' the features and vectorize the data
    vectors = vectorizer.fit_transform(data.data)

    print(vectors.toarray())  # print numpy vectors
    # %%
    # Linear SVM model
    model = LinearSVC(C=0.0001)

    # cross-validation evaluation
    for metric_score in ['f1_macro', 'f1_micro', 'f1_weighted']:
        scores = cross_validate(model, vectors, data.target, cv=stratified_k_fold, scoring=[metric_score])
        print('\033[1m', metric_score.upper(), '\033[0m', ':',
              round(sum(scores['test_' + metric_score]) / len(scores['test_' + metric_score]), 2))

    ## 2.6 without lowercasing (NoLowercase)
    print("\n2.6 without lowercasing (NoLowercase)")
    ### 2.6.1 without lowercasing (NoLowercase) with Count Vectorization (CountVect)
    print("\n2.6.1 without lowercasing (NoLowercase) with Count Vectorization (CountVect)")
    # Count Vectorization without lowercasing
    vectorizer = CountVectorizer(lowercase=False)

    # use fit_transform to 'learn' the features and vectorize the data
    vectors = vectorizer.fit_transform(data.data)

    # Linear SVM model
    model = LinearSVC(C=0.0001)

    # cross-validation evaluation
    for metric_score in ['f1_macro', 'f1_micro', 'f1_weighted']:
        scores = cross_validate(model, vectors, data.target, cv=stratified_k_fold, scoring=[metric_score])
        print('\033[1m', metric_score.upper(), '\033[0m', ':',
              round(sum(scores['test_' + metric_score]) / len(scores['test_' + metric_score]), 2))
    ### 2.6.2 without lowercasing (NoLowercase) with TF-IDF Transformation (TF-IDF)
    print("\n2.6.2 without lowercasing (NoLowercase) with TF-IDF Transformation (TF-IDF)")
    # TF-IDF Transformation without lowercasing
    transformer = TfidfTransformer()

    # use fit_transform to 'learn' the features and vectorize the data
    vectors = transformer.fit_transform(vectors)
    # Linear SVM model
    model = LinearSVC(C=0.0001)

    # cross-validation evaluation
    for metric_score in ['f1_macro', 'f1_micro', 'f1_weighted']:
        scores = cross_validate(model, vectors, data.target, cv=stratified_k_fold, scoring=[metric_score])
        print('\033[1m', metric_score.upper(), '\033[0m', ':',
              round(sum(scores['test_' + metric_score]) / len(scores['test_' + metric_score]), 2))

    ### 2.6.3 without lowercasing (NoLowercase) with TF-IDF Vectorization (TFIDF)
    print("\n2.6.3 without lowercasing (NoLowercase) with TF-IDF Vectorization (TFIDF)")
    # TF-IDF Vectorization without lower casing
    vectorizer = TfidfVectorizer(lowercase=False)

    # use fit_transform to 'learn' the features and vectorize the data
    vectors = vectorizer.fit_transform(data.data)

    # Linear SVM model
    model = LinearSVC(C=0.0001)

    # cross-validation evaluation
    for metric_score in ['f1_macro', 'f1_micro', 'f1_weighted']:
        scores = cross_validate(model, vectors, data.target, cv=stratified_k_fold, scoring=[metric_score])
        print('\033[1m', metric_score.upper(), '\033[0m', ':',
              round(sum(scores['test_' + metric_score]) / len(scores['test_' + metric_score]), 2))
