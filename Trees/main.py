import sys
import string
import pandas as pd
import numpy as np
from random import randrange

from pre_proc import *
from decision_tree_class import *
from random_forest import *
from boost_2 import *
from bagging import *


def model_select_train_test(train_file_name,test_file_name,mdl):
    max_depth = 10
    min_size = 10
    num_trees  = 50
    function_to_apply = 1
    num_feat = 2
    
    w = 1000
    
    df_train = pd.read_csv(train_file_name, delimiter='\t|\n', header=None, names=['id', 'class_label_naive', 'reviewText'],
                           engine='python')
    df_test = pd.read_csv(test_file_name, delimiter='\t|\n', header=None, names=['id', 'class_label_naive', 'reviewText'],
                          engine='python')
    
    df_train.reviewText = df_train.reviewText.apply(fnc)
    df_temp_wrds_slt = df_train.copy()
    
    df_train.reviewText = df_train.reviewText.apply(fnc_vectorize(function_to_apply))
    df_temp_wrds_slt.reviewText = df_temp_wrds_slt.reviewText.apply(fnc_vectorize(1))
    
    df_test.reviewText = df_test.reviewText.apply(fnc)
    df_test.reviewText = df_test.reviewText.apply(fnc_vectorize(function_to_apply))
    
    #_, _, key_val = making_bag_of_features(df_train, w, True)
    wrds_slt = frequent_words(df_temp_wrds_slt, w)
    df_feat_train = bag_of_feature(df_train, wrds_slt,add_class = True)
    df_feat_test = bag_of_feature(df_test, wrds_slt,add_class = True)
    
    X_train,Y_train,X_test,Y_test,XY_train,XY_test = dataframe_to_numpy(df_feat_train,df_feat_test)
    
    if mdl == 1:
        err_dec = decision_fnc(XY_train.copy(),XY_test.copy(),max_depth=max_depth,min_size=min_size)
        print 'ZERO-ONE-LOSS-DECISION TREE {}'.format(err_dec)
    elif mdl == 2:
        err_bag = bagging_func(XY_train.copy(),XY_test.copy(),max_depth=max_depth,min_size=min_size,num_trees=num_trees)
        print 'ZERO-ONE-LOSS-BAGGING {}'.format(err_bag)
    elif mdl ==3:
        err_rand = random_fnc(XY_train.copy(),XY_test.copy(),max_depth=max_depth,min_size=min_size,num_trees=num_trees)
        print 'ZERO-ONE-LOSS-RANDOM FOREST {}'.format(err_rand)
    elif mdl ==4:
        err_boost = boosting_function(X_train.copy(),Y_train.copy(),X_test.copy(),Y_test.copy(),max_depth=max_depth,min_size=min_size,num_trees=num_trees)
        print 'ZERO-ONE-LOSS-BOOST {}'.format(err_boost)
    else:
        print('Wrong Model Number')
    return 0


#%%
def main():
    #print('hello')
    num_imputs = len(sys.argv[1:])
    inputs = sys.argv[1:]
    #print(num_imputs)
    mdl = int(inputs[2])

    model_select_train_test(inputs[0],inputs[1],mdl)
 
if __name__ == "__main__":
    main()  

