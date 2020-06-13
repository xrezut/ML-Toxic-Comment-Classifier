# Yaaseen Asmal (U1652830)
# University of Huddersfield
# Computer Science
# Final Year Project
# 12/05/2020
# Logistic Regression Pipeline

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import nltk
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn import datasets, tree
from sklearn.metrics import confusion_matrix, log_loss, accuracy_score, hamming_loss, f1_score, precision_score, recall_score, classification_report
from six import StringIO
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer


print("######################################")
print("Starting Script....")
print("######################################")

#Read the csv file into dataframe df and print so we can see
print("Reading File....")
df = pd.read_csv("train.csv")
print(df.shape)
print("File Read")

print("######################################")
print("Seperating comments and labels....")

#Below line causes shuffling of data, so train_test_split method is not needed after
df = df.reindex(np.random.permutation(df.index))

comment = df['comment_text']
comment = comment.as_matrix()

label = df[['toxic']]
label = label.as_matrix()

print("######################################")

comments = []
labels = []

comments = comment
labels = label

labels = np.asarray(labels)

print("######################################")

print("Making punctiation string....")
#Preparing a string containing all punctuations to be removed
punctuation_edit = string.punctuation.replace('\'','') +"0123456789"
outtab = "                                         "
trantab = str.maketrans(punctuation_edit, outtab)

print("Making stopwords list...")
#remove stop words and any single letters produced from other pre processing stages
stop_words = stopwords.words('english')
stop_words.append('')
stop_words.remove('not')
for x in range(ord('b'), ord('z')+1):
    stop_words.append(chr(x))

print("######################################")
print("Starting text preprocessing....")
#stemming and lemmatisation
#create objects for stemmer and lemmatizer
lemmatiser = WordNetLemmatizer()
stemmer = PorterStemmer()

#loop through all comments and apply the preprocessing. 
# remove punctuation -> tokenize -> stemming -> lemmatization -> join words again together to reprocess
for i in range(len(comments)):
    comments[i] = comments[i].lower().translate(trantab)
    l = []
    for word in comments[i].split():
        l.append(stemmer.stem(lemmatiser.lemmatize(word,pos="v")))
    comments[i] = " ".join(l)
print("Preprocessing completed for all comments")

print("######################################")
print("Applying count Vectorizer....")
##Applying TF-IDF
#create object supplying our custom stop words
tfidfconverter = TfidfVectorizer(max_features=5000, min_df=5, max_df=0.7, stop_words=stop_words)
#fitting it to converts comments into TF-IDF format
tf = tfidfconverter.fit_transform(comments).toarray()

# print(count_vector.get_feature_names())
print("Printing tf shape....")
print(tf.shape)

print("Count Vectorizer completed")
print("######################################")


# use half the data from the dataset for testing and 1/3 (.3) for training
# X = features and Y = labels
# X_train = features training data -- X_test = features test data
# y_train = label training data -- y_test = label test data
# custom shuffle method does job of train test split
def shuffle(matrix, target, test_proportion):
    ratio = int(matrix.shape[0]/test_proportion)
    X_train = matrix[ratio:,:]
    X_test =  matrix[:ratio,:]
    Y_train = target[ratio:,:]
    Y_test =  target[:ratio,:]
    return X_train, X_test, Y_train, Y_test

X_train, X_test, Y_train, Y_test = shuffle(tf, labels,3)


print("Training classifier....")
# create Decision Tree Classifier
my_classifier = LogisticRegression()

# train the classifier using the training data
my_classifier.fit(X_train, Y_train)

print("Classifier trained")

# classify and our testing data
predictions = my_classifier.predict(X_test)

print("######################################")


print("Evaluation Metrics")

print("Hamming Loss: {}".format(hamming_loss(Y_test, predictions)*100))
print("Accuracy: {}".format(accuracy_score(Y_test, predictions)*100))
print("Log Loss: {}".format(log_loss(Y_test, predictions)))

print("F1 Score: {}".format(f1_score(Y_test, predictions, average="weighted")))
print("Precision: {}".format(precision_score(Y_test, predictions, average="weighted")))
print("Recall: {}".format(recall_score(Y_test, predictions, average="weighted")))

print("######################################")
print("######################################")
print("######################################")
print("######################################")

#save the model so we can use later and dont need to train again
print("Saving model....")
with open('comment_classifier.pkl', 'wb') as fout:
    pickle.dump((tfidfconverter, my_classifier), fout)
print("Model saved")

print("######################################")
print("######################################")
print("Script complete")