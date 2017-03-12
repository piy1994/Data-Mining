YELP DATASET REVIEW CLASSIFICATION: NAIYE BAYES

README
NAME: PIYUSH DUGAR
Python Version : Python 2.7.12 

Libraries used : 
sys
string
pandas
numpy

This programme needs pandas to run.

1. The main file is nbc.py
2. The file can take 2 and 1 inputs
3. If 2 inputs are given, then training and testing will run and it will print out the top 10 frequecy words with ties and zero-one-loss
4. Please run it as "python nbc.py training_file_name.csv testing_file_name.csv " to run the code. Here first input is path to the training file and second input is path to the testing file.

5. If you just input one path , i.e something like this "python nbc.py [file_name.csv]" then it will run traingn and testing on random 50% of the data and will output 4 matrices: 1. error on naive bayes with varying percentage 2. baseline error on varying percentage 3. error on naive bayes with varying number of words 4. base line error on varying words.