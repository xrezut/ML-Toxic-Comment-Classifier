# Yaaseen Asmal (U1652830)
# University of Huddersfield
# Computer Science
# Final Year Project
# 12/05/2020

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
from sklearn.metrics import accuracy_score
from six import StringIO
import pydot
import pickle

filename = 'comment_classifier.pkl'

print("######################################")
print("Starting Message Classifier. Enter quit at anytime to exit....")
print("######################################")

userInput = ""
while userInput != 'quit':
    # Take user input 
    print("Input your text: ")
    userInput = input() 
    print("######################################")
    if userInput == 'quit':
        quit()

    commentsInput = [userInput]

    # Preparing a string containing all punctuations to be removed
    punctuation_edit = string.punctuation.replace('\'','') +"0123456789"
    outtab = "                                         "
    trantab = str.maketrans(punctuation_edit, outtab)

    # Make stopwords list
    stop_words = stopwords.words('english')
    stop_words.append('')
    stop_words.remove('not')
    for x in range(ord('b'), ord('z')+1):
        stop_words.append(chr(x))

    # Create objects for stemmer and lemmatizer
    lemmatiser = WordNetLemmatizer()
    stemmer = PorterStemmer()

    # Loop through all comments and apply the preprocessing. Stores values in comments variable.
    # Remove punctuation -> tokenize -> stemming -> lemmatization -> join words again together to reprocess
    for i in range(len(commentsInput)):
        commentsInput[i] = commentsInput[i].lower().translate(trantab)
        l = []
        for word in commentsInput[i].split():
            l.append(stemmer.stem(lemmatiser.lemmatize(word,pos="v")))
        commentsInput[i] = " ".join(l)

    # read and load model file
    with open(filename, 'rb') as f:
        cVec, model = pickle.load(f)
    results = cVec.transform(commentsInput).toarray()

    # make predictions on the user input
    predictionsNew = model.predict(results)

    a = ["Toxic"]
    b = np.ndarray((1,), buffer=np.array(predictionsNew), dtype=int) 

    res = 0;
    for x, y in zip(a, b):
        print(f"{x} = {y}")
        res = y

    choice = 0

    # message not quarantined
    if res == 0:
        print("The input is safe...")
        print("Your message was: " + userInput)

    # message quarantined
    if res == 1:
        print("The input is possibly unsafe...")
        print(res)
        print("Content quarantined. Would you like to view the content? Enter Y or N....")
        choice = input().lower()

    # allow user to decide if they want to view or delete the qurantined message    
    yes = {'yes','y', 'ye', ''}
    no = {'no','n'}
    if choice in yes:
       print("Your message was: " +userInput)
    elif choice in no:
       print("The content has been deleted.")
    
    print("######################################")
    print("Classification Done")
    print("######################################")