# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 23:29:27 2017

@author: piyush
"""
import sys
import string
import pandas as pd
import numpy as np


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
            lst.append(1)
        else:
            lst.append(0)
    return lst


def making_bag_of_features(df, w, print_top_10=False):
    # df = pd.read_csv(df_path,delimiter='	', header=None,names = ['id', 'label', 'reviewText'])

    main_dict = {}
    for rev in df.reviewText:
        my_dict = rev
        main_dict = dict_update_1(my_dict, main_dict)

    key_val = remove_top_100(main_dict)
    key_val = key_val[100:100 + w]
    # print(key_val)
    cols = [label[0] for label in key_val]
    megalist = []
    for rev in df.reviewText:
        lst = dependent_var(rev, cols)
        megalist.append(lst)
    df1 = pd.DataFrame(megalist, columns=cols)

    df1.insert(0, 'class_label_naive', df['class_label_naive'])
    # df1.to_csv('feature_table.csv')
    if print_top_10:
        return df1, cols, key_val
    else:
        return df1, cols

def naive_bayes_clsf(df):
    # print('new naive bayes')
    # df = pd.read_csv('feature_table.csv')
    df_p = df[df['class_label_naive'] == 1]
    df_n = df[df['class_label_naive'] == 0]
    df_p = df_p.drop(df.columns[0], axis=1)
    df_n = df_n.drop(df.columns[0], axis=1)
    freq_p = df_p.sum(axis=0)
    freq_n = df_n.sum(axis=0)
    cfd = np.array([freq_p, freq_n])
    p_y_n = np.array([float(df_p.shape[0]) / float(df.shape[0]), float(df_n.shape[0]) / float(df.shape[0])])
    cpd = np.zeros((4, df_p.shape[1]), dtype=float)

    # cpd[0,:] =np.divide(cfd[0,:],float(df_p.shape[0]))
    cpd[0, :] = np.divide(cfd[0, :] + 1.0, float(df_p.shape[0]) + 2.0)
    # cpd[0,:][cpd[0,:]==0] = 1/(float(df_p.shape[0])+2.0)
    # cpd[0,:][cpd[0,:]==0] = 1/(float(df_p.shape[0])+df_p.shape[1])

    cpd[1, :] = 1 - cpd[0, :]
    cpd[1, :][cpd[1, :] == 0] = 1 / (float(df_p.shape[0]) + 2.0)
    # cpd[1,:][cpd[1,:]==0] = 1/(float(df_p.shape[0])+df_p.shape[1])

    # cpd[2,:] = np.divide(cfd[1,:],float(df_n.shape[0]))
    cpd[2, :] = np.divide(cfd[1, :] + 1.0, float(df_n.shape[0]) + 2.0)
    # cpd[2,:][cpd[2,:]==0] = 1/(float(df_n.shape[0])+2.0)
    # cpd[2,:][cpd[2,:]==0] = 1/(float(df_n.shape[0])+df_n.shape[1])

    cpd[3, :] = 1 - cpd[2, :]
    cpd[3, :][cpd[3, :] == 0] = 1 / (float(df_n.shape[0]) + 2.0)
    # cpd[3,:][cpd[3,:]==0] = 1/(float(df_n.shape[0])+df_n.shape[1])

    return cpd, p_y_n


def train(df_train, w):
    df_feat, wrds_slt = making_bag_of_features(df_train, w)
    cpd, p_y_n = naive_bayes_clsf(df_feat)
    return cpd, p_y_n, wrds_slt


def bag_of_feature_for_test(df_test, wrds_slt):
    df1 = pd.DataFrame(data=None, index=None, columns=wrds_slt)
    megalist = []
    for rev in df_test.reviewText:
        lst = dependent_var(rev, wrds_slt)
        megalist.append(lst)
    df1 = pd.DataFrame(megalist, columns=wrds_slt)
    return df1


def error(orig_label, predict_label):
    orig_label = np.array(orig_label)
    predict_label = np.array(predict_label)
    diff = (predict_label != orig_label)
    err = float(np.sum(diff)) / float(len(orig_label))
    return err


def prob_yes_no(features, cpd, p_y_n):
    p_num_y = float(1)
    i = 0
    for feat in features:
        if feat == 1:
            p_num_y *= cpd[0, i]
        else:
            p_num_y *= cpd[1, i]
        i += 1

    p_num_y *= p_y_n[0]

    p_denum_n = float(1)
    i = 0
    for feat in features:
        if feat == 1:
            p_denum_n *= cpd[2, i]
        else:
            p_denum_n *= cpd[3, i]
        i += 1
    p_denum_n *= p_y_n[1]
    p_yes = p_num_y / (p_num_y + p_denum_n)
    return p_yes


def test(df_test, cpd, p_y_n, wrds_slt):
    df_feat_test = bag_of_feature_for_test(df_test, wrds_slt)
    orig_label = list(df_test['class_label_naive'])
    predicted_label = []
    for i in range(df_feat_test.shape[0]):
        features = list(df_feat_test.iloc[i])
        p_yes = prob_yes_no(features, cpd, p_y_n)
        # print(p_yes)
        if p_yes >= 0.5:
            predicted_label.append(1)
        else:
            predicted_label.append(0)
    err = error(orig_label, predicted_label)
    # return err,predicted_label
    return err, orig_label


def q1_q2(file_path1, file_path2):
    fnc = lambda x: x.lower().translate(None, string.punctuation).split()
    fnc2 = lambda x: dict([(word, 1) for word in x])
    df_train = pd.read_csv(file_path1, delimiter='\t|\n', header=None, names=['id', 'class_label_naive', 'reviewText'],
                           engine='python')
    df_test = pd.read_csv(file_path2, delimiter='\t|\n', header=None, names=['id', 'class_label_naive', 'reviewText'],
                          engine='python')

    df_train.reviewText = df_train.reviewText.apply(fnc)
    df_train.reviewText = df_train.reviewText.apply(fnc2)

    df_test.reviewText = df_test.reviewText.apply(fnc)
    df_test.reviewText = df_test.reviewText.apply(fnc2)

    w = 500
    _, _, key_val = making_bag_of_features(df_train, w, True)
    cpd, p_y_n, wrds_slt = train(df_train, w)
    err, _ = test(df_test, cpd, p_y_n, wrds_slt)
    print_words = printing_words(key_val)
    return err, wrds_slt, key_val, print_words


def q3_q4(file_path):
    fnc = lambda x: x.lower().translate(None, string.punctuation).split()
    fnc2 = lambda x: dict([(word, 1) for word in x])
    df = pd.read_csv(file_path, delimiter='\t|\n', header=None, names=['id', 'class_label_naive', 'reviewText'],
                     engine='python')
    # df = pd.read_csv("yelp_data.csv",delimiter='	', header=None,names = ['id', 'class_label_naive', 'reviewText'])
    df.reviewText = df.reviewText.apply(fnc)
    df.reviewText = df.reviewText.apply(fnc2)
    err_array_q3, err_array_beseline_q3 = q3(df)
    err_array_q4, err_array_beseline_q4 = q4(df)
    return err_array_q3, err_array_beseline_q3, err_array_q4, err_array_beseline_q4


def q3(df):
    w = 500
    err_array = []
    err_array_baseline = []
    for percent in [1.0, 5.0, 10.0, 20.0, 50.0, 90.0]:

        err_each_percent = []
        err_each_percent_baseline = []
        for i in range(10):
            df_train = df.sample(frac=percent / 100.0)
            df_test = df.drop(df_train.index)

            df_train = df_train.reset_index(drop=True)
            df_test = df_test.reset_index(drop=True)
            # print(df_train.shape[0])
            cpd, p_y_n, wrds_slt = train(df_train, w)
            err, orig_label = test(df_test, cpd, p_y_n, wrds_slt)
            err_each_percent.append(err)

            baseline_pred = [1 if p_y_n[0] >= p_y_n[1] else 0]
            err_baseline = error(orig_label, baseline_pred * len(orig_label))
            err_each_percent_baseline.append(err_baseline)
            # if err<err_min:
            #   err_min = err
        # print(err_min)
        err_array.append(err_each_percent)
        err_array_baseline.append(err_each_percent_baseline)
    return np.array(err_array), np.array(err_array_baseline)


def q4(df):
    err_array = []
    err_array_baseline = []
    for w in [10, 50, 250, 500, 1000, 4000]:

        err_each_w = []
        err_each_w_baseline = []
        for i in range(10):
            df_train = df.sample(frac=0.5)
            df_test = df.drop(df_train.index)

            df_train = df_train.reset_index(drop=True)
            df_test = df_test.reset_index(drop=True)
            # print(df_train.shape[0])
            cpd, p_y_n, wrds_slt = train(df_train, w)
            err, orig_label = test(df_test, cpd, p_y_n, wrds_slt)
            err_each_w.append(err)

            baseline_pred = [1 if p_y_n[0] >= p_y_n[1] else 0]
            err_baseline = error(orig_label, baseline_pred * len(orig_label))
            err_each_w_baseline.append(err_baseline)
            # if err<err_min:
            #   err_min = err
        # print(err_min)
        err_array.append(err_each_w)
        err_array_baseline.append(err_each_w_baseline)
    return np.array(err_array), np.array(err_array_baseline)


def printing_words(key_val):
    main_lst = []
    j = 0
    l = 0
    while l < 10:
        lst = []
        while j < range(len(key_val)):
            if key_val[j + 1][1] == key_val[j][1]:

                lst.append(key_val[j][0])
                j += 1
            else:
                lst.append(key_val[j][0])
                main_lst.append(lst)
                j += 1
                l += 1
                break
    return main_lst


def print_the_words(print_words, err):
    i = 0
    for lst in print_words:
        for word in lst:
            print(' '.join(['WORD'] + [str(i + 1)] + [word]))
        i += 1
    print(' '.join(['ZERO-ONE-LOSS'] + [str(err)]))


def main():
    num_imputs = len(sys.argv[1:])
    inputs = sys.argv[1:]
    #print(inputs)
    #print(sys.argv)
    if num_imputs == 2:
        err,wrds_slt,key_val,print_words =q1_q2(inputs[0],inputs[1])
        #print(err)
        print_the_words(print_words,err)
    else:
        err_array_q3,err_array_beseline_q3,err_array_q4,err_array_beseline_q4 = q3_q4(inputs[0])
        print('naive bayes error array for q3')
        print(err_array_q3)
        print('baseline error array for q3')
        print(err_array_beseline_q3)
        print('naive bayes error array for q4')
        print(err_array_q4)
        print('baseline error array for q4')
        print(err_array_beseline_q4)
        
if __name__ == "__main__":
    main()
