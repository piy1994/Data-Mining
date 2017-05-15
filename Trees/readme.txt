YELP DATASET REVIEW CLASSIFICATION ON:
1. Decision Tree
2. Bagging Trees
3. Random Forest
4. Boosting

The program outputs the zero one loss
A demo file, yelp_data_train and yelp_data_test has been included

Specification:
Python Version : Python 2.7.12 

Libraries used : 
sys
string
pandas
numpy
random


1.The main file is main.py
2. It needs 3 inputs to run : "training file path" "testing file path" "model idx" 
3. Please run it as "python main.py training_file_name.csv testing_file_name.csv " "model_index" to run the code. Here first input is path to the training file and second input is path to the testing file and third input is which model to run (1 for decision tree,2 for Bagging Tree ,3 for Random Forest and 4 for boosting)
4. If there are any runtime error, please append python -W 
"python -W ignore main.py train_data test_data 2"
5. The default parameters are :
    maximum depth = 10
    minimum size of a node of tree = 10
    number of trees = 50
    You can change them in the function model_select_train_test()
