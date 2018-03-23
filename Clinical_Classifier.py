import xml.etree.ElementTree as ElementTree
from nltk import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from textblob import TextBlob
import re
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer, LabelBinarizer
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import tree
from sklearn import linear_model
from sklearn.multiclass import OneVsRestClassifier

stop_words = stopwords.words("english")
cachedStopWords = stopwords.words("english")
ngrams_number = 3

def tokenize(text):
    min_length = 3

    words = map(lambda word: word.lower(), word_tokenize(text))
    #   words = list(map(lambda word: word.lower(), word_tokenize(text)))

    words = [word for word in words if word not in cachedStopWords]
    tokens = (list(map(lambda token: PorterStemmer().stem(token), words)))
    p = re.compile('[a-zA-Z]+')
    filtered_tokens = list(filter(lambda token: p.match(token) and
                                                len(token) >= min_length,
                                  tokens))
    filtered_tokens = ' '.join(filtered_tokens)
    filtered_tokens = [' '.join(ngram) for ngram in TextBlob(filtered_tokens).ngrams(ngrams_number)]

    return filtered_tokens

############################################################################################################
# Parsing xml file to string and convert to txt files
def parse(file):
    # Parsing xml contents to string
    with open("C:/Users/Mahdi/Desktop/data/" + file, 'r') as myfile:
        content = myfile.read()
        # content = myfile.read().replace('\n', '')
    root = ElementTree.fromstring(content)

    # Searching for the target tags to get the data
    for log in root.iter('TEXT'):
        data = log.text

    # Flushing out the contents to txt files
    with open("C:/Users/Mahdi/Desktop/converted_data/" + file[:-4] + ".txt", "w") as mynewfile:
        mynewfile.write(data)
    return data


if __name__ == '__main__':
    ########################################
    import fnmatch
    import os

    FL = [f for f in os.listdir("C:/Users/Mahdi/Desktop/data/") if fnmatch.fnmatch(f, '*.xml')]

    # Saving the context of all documents in 'train_docs' list
    train_docs = []
    test_docs = []

    main_docs = [parse(FL[i]) for i in range(len(FL))]

    # Producing random label between 0 and 1 for the documents
    train_labels = []
    test_labels = []

    main_labels = [random.randint(0, 1) for i in range(len(main_docs))]

    split_index = len(main_docs) - int(len(main_docs) * 30 / 100)

    # Spliting 70% of data set for training and 30% for testing
    train_labels = main_labels[:split_index]
    test_labels = main_labels[split_index:]

    train_docs = main_docs[:split_index]
    test_docs = main_docs[split_index:]

    # Tokenisation
    vectorizer = TfidfVectorizer(stop_words=stop_words, tokenizer=tokenize)
    vectorised_train_documents = vectorizer.fit_transform(train_docs)
    vectorised_test_documents = vectorizer.transform(test_docs)

# ###################### classifier  #######################
classifier = OneVsRestClassifier(LinearSVC(random_state=42))
# classifier = MultinomialNB(fit_prior=True)
# classifier = MultinomialNB()
# classifier = LinearSVC(random_state=42)
# classifier = LinearSVC()
# classifier = GaussianNB()        Failed
# classifier = KNeighborsClassifier()
# classifier = tree.DecisionTreeClassifier()
# classifier =  linear_model.LogisticRegression(C=1e5)
# classifier =  linear_model.LogisticRegression()

###############################################
classifier.fit(vectorised_train_documents, train_labels)
predictions = classifier.predict(vectorised_test_documents)

########################################
from sklearn.metrics import f1_score, precision_score, recall_score

precision = precision_score(test_labels, predictions, average='micro')
recall = recall_score(test_labels, predictions, average='micro')
f1 = f1_score(test_labels, predictions, average='micro')

print("Micro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))

precision = precision_score(test_labels, predictions, average='macro')
recall = recall_score(test_labels, predictions, average='macro')
f1 = f1_score(test_labels, predictions, average='macro')

print("Macro-average quality numbers")
print("Precision: {:.4f}, Recall: {:.4f}, F1-measure: {:.4f}".format(precision, recall, f1))
########################################