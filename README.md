# ML-Toxic-Comment-Classifier
This repo contains the models developed for my thesis, to classify toxic comments using Supervised Learning.

NOTES:
If you would like to train the model, or test then you must first need to download the necessary python supporting libraries
for your computer. Also the dataset should be in the same directory as the location of the trainModel.py file.

To save time the environment.yml can be used to setup the necessary libraries in your environment using Anaconda.
Additionally, you will also need to download the nltk datafiles. 



This repo contains the supporting code, and related files which have been developed 
and used throughout this project. See below for what each folder contains.

1_SVM Pipeline
=======================
Contains the python file which has the code that was developed to train the model. 
Contains the comment_classifier.pkl (zipped in folder due to github restriction) file which contains the saved model that was trained during development.


2_Qurantining Framework Script
=======================
Contains the python script file which is a program that prototypes and shows the qurantining framework. It uses
the trained SVM model from the comment_classifier.pkl file to classify data.

3_Dataset
=======================
Contains a CSV file of the dataset that was used for training the models.


4_Logistic Regression
5_Decision Tree
6_Naive Bayes
=======================
Each folder in these directory contains:
The python file which has the code that was developed to train the model. 
The comment_classifier.pkl file which contains the saved model that was trained during development.
These algorithms were trained and analysed, but were not used in the final conclusion since the accuracy of the SVM was better.

