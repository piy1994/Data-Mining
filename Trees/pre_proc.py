import sys
import string
import pandas as pd
import numpy as np


fnc = lambda x: x.lower().translate(None, string.punctuation).split()
fnc_one = lambda x: dict([(word, 1) for word in x])
def fnc_two(lst):
    dic = {}
    for word in lst:
        if word in dic:
            dic[word] = 2
        else:
            dic[word] = 1
    return dic
def fnc_vectorize(function_to_apply):
    if function_to_apply==1:
        return fnc_one
    else:
        return fnc_two



def dict_update_1(my_dict, main_dict):
    key_my_dict = my_dict.keys()
    for item in key_my_dict:
        if item in main_dict:
            main_dict[item] += my_dict[item]
        else:
            main_dict[item] = my_dict[item]
    return main_dict


# function to remove top 100
def remove_top_100(main_dict):
    key_val = zip(main_dict.keys(), main_dict.values())
    key_val.sort(key=lambda x: x[1], reverse=True)
    # return key_val[101:101+w]
    return key_val


def dependent_var(my_dict, cols):
    lst = []
    for word in cols:
        if word in my_dict:
            lst.append(my_dict[word])
        else:
            lst.append(0)
    return lst


def frequent_words(df, w):
    # df = pd.read_csv(df_path,delimiter='	', header=None,names = ['id', 'label', 'reviewText'])

    main_dict = {}
    for rev in df.reviewText:
        my_dict = rev
        main_dict = dict_update_1(my_dict, main_dict)

    key_val = remove_top_100(main_dict)
    key_val = key_val[100:100 + w]
    # print(key_val)
    cols = [label[0] for label in key_val]
    return cols



def bag_of_feature(df, wrds_slt,add_class = False):
    df1 = pd.DataFrame(data=None, index=None, columns=wrds_slt)
    megalist = []
    for rev in df.reviewText:
        lst = dependent_var(rev, wrds_slt)
        megalist.append(lst)
    df1 = pd.DataFrame(megalist, columns=wrds_slt)
    if add_class:
        df1.insert(0, 'class_label_naive', list(df['class_label_naive']))
    return df1


def error(orig_label, predict_label):
    if len(orig_label)!=len(predict_label):
        print('Dimension not equla')
        return
    orig_label = np.array(orig_label)
    predict_label = np.array(predict_label)
    diff = (predict_label != orig_label)
    err = float(np.sum(diff)) / float(len(orig_label))
    return err
    

def dataframe_to_numpy(df_feat_train_input,df_feat_test_input):
    df_feat_train = df_feat_train_input.copy()
    df_feat_test = df_feat_test_input.copy()
    X_train = np.array(df_feat_train.drop('class_label_naive',axis=1))
    Y_train = np.array((df_feat_train.class_label_naive)).reshape((X_train.shape[0],1))
    
    X_test = np.array(df_feat_test.drop('class_label_naive',axis=1))
    Y_test = np.array((df_feat_test.class_label_naive)).reshape((X_test.shape[0],1))
    
    XY_train = np.hstack((X_train,Y_train))
    XY_test = np.hstack((X_test,Y_test))
    return X_train,Y_train,X_test,Y_test,XY_train,XY_test
