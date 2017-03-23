import numpy as np
import pandas as pd
import sys
import string


#%% vectorizing functions
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


#%% NBC complete


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

def naive_bayes_clsf(df,num_feat):
    # print('new naive bayes')
    # df = pd.read_csv('feature_table.csv')
    df_p = df[df['class_label_naive'] == 1]
    df_n = df[df['class_label_naive'] == 0]
    df_p = df_p.drop(df.columns[0], axis=1)
    df_n = df_n.drop(df.columns[0], axis=1)
    
    class_label= range(num_feat)
    freq_p = []
    freq_n = []
    
    class_count_p = df_p.apply(pd.value_counts)
    class_count_n = df_n.apply(pd.value_counts)
    for cls in class_label:
        freq = list(class_count_p.iloc[cls])
        freq_p.append(freq)
        freq = list(class_count_n.iloc[cls])
        freq_n.append(freq)
    
    #freq_p = df_p.sum(axis=0)
    #freq_n = df_n.sum(axis=0)
    cfd = np.vstack([freq_p, freq_n])
    cfd = np.nan_to_num(cfd)
    p_y_n = np.array([float(df_p.shape[0]) / float(df.shape[0]), float(df_n.shape[0]) / float(df.shape[0])])
    cpd = np.zeros((2*num_feat, df_p.shape[1]), dtype=float)
    for i in range(num_feat):
        cpd[i, :] = np.divide(cfd[i, :] + 1.0, float(df_p.shape[0]) + float(num_feat))
        cpd[i+num_feat,:] = np.divide(cfd[i+num_feat, :] + 1.0, float(df_n.shape[0]) + float(num_feat)) 
    return cpd, p_y_n,cfd



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
    orig_label = np.array(orig_label)
    predict_label = np.array(predict_label)
    diff = (predict_label != orig_label)
    err = float(np.sum(diff)) / float(len(orig_label))
    return err


def prob_yes_no(features, cpd, p_y_n,num_feat = 2):
    p_num_y = float(1)
    i = 0
    feature_numbers = range(num_feat)
    for feat in features:
        for feat_num in feature_numbers:
            if feat == feat_num:
                p_num_y*=cpd[feat_num,i]
                i+=1
    

    p_num_y *= p_y_n[0]

    p_denum_n = float(1)
    i = 0
    
    for feat in features:
        for feat_num in feature_numbers:
            if feat == feat_num:
                p_denum_n *= cpd[feat_num + num_feat, i]
                i+=1
    p_denum_n *= p_y_n[1]
    p_yes = p_num_y / (p_num_y + p_denum_n)
    return p_yes


def test(df_feat_test, cpd, p_y_n, wrds_slt,num_feat):
    orig_label = list(df_feat_test.class_label_naive)
    df_feat_test = df_feat_test.drop('class_label_naive',axis=1)
    predicted_label = []
    for i in range(df_feat_test.shape[0]):
        features = list(df_feat_test.iloc[i])
        p_yes = prob_yes_no(features, cpd, p_y_n,num_feat)
        # print(p_yes)
        if p_yes >= 0.5:
            predicted_label.append(1)
        else:
            predicted_label.append(0)
    err = error(orig_label, predicted_label)
    # return err,predicted_label
    return err, orig_label
    
    
def nbc_combined(df_feat_train,num_feat,df_feat_test,wrds_slt):
    cpd, p_y_n,_ = naive_bayes_clsf(df_feat_train,num_feat)
    err,_ = test(df_feat_test,cpd,p_y_n,wrds_slt,num_feat)
    return err

#%% Logistics regression

def sigmoid(prod_feat_wt):
    return 1/(1+np.exp(-prod_feat_wt))
    
def gradient_lr(X,W,Y,lambd):
    m = float(Y.shape[0])
    #print(m)
    W_1 = W.copy()
    W_1[0] = 0
    h = sigmoid(np.dot(X,W))
    #print(h)
    p = (lambd*np.dot(W_1.T,W_1))/float(2*m);
    #p = (lambd*np.dot(W_1.T,W_1))/float(2);
    J =(1/m)*(np.sum(np.multiply(-Y,np.log(h)) - np.multiply(1-Y,np.log(1-h)))) + p
    #J = (np.sum(np.multiply(-Y,np.log(h)) - np.multiply(1-Y,np.log(1-h)))) + p    
    #grad = (np.dot(X.T,(h - Y)) + lambd*W_1)/m
    grad = (np.dot(X.T,(h - Y)) + lambd*W_1)
    return grad,J
    


def logistics_regression(df_train,df_test):
    X_train = np.array(df_train.drop('class_label_naive',axis=1))
    intercept = np.ones((X_train.shape[0],1))
    X_train = np.hstack((intercept,np.array(X_train)))
    Wt = np.zeros((X_train.shape[1],1))
    
    
    Y_train = np.array(df_train['class_label_naive'])
    Y_train = Y_train.reshape(Y_train.shape[0],1)
    num_steps = 100
    err_arr = []
    lambd = 0.01
    alpha = 0.01
    thresh = 10**-6
    for step in range(num_steps):
        grad,err = gradient_lr(X_train,Wt,Y_train,lambd)
        Wt_new = Wt - alpha*grad
        if np.linalg.norm(Wt_new-Wt)<thresh:
            break
        Wt = Wt - alpha*grad
        err_arr.append(err)
    #print(min(err_arr))
    
    #now testing
    X_test = np.array(df_test.drop('class_label_naive',axis=1))
    intercept = np.ones((X_test.shape[0],1))
    X_test = np.hstack((intercept,np.array(X_test)))
    
    Y_test = np.array(df_test['class_label_naive'])
    Y_test = Y_test.reshape(Y_test.shape[0],1)
    
    Y_cap = sigmoid(np.dot(X_test,Wt))
    Y_cap[Y_cap >=0.5] = 1
    Y_cap[Y_cap <0.5] = 0
    #print(np.sum(Y_cap == Y_test)/float(Y_cap.shape[0]))
    #return err_arr
    return(error(Y_test, Y_cap))

#%% Support Vector

def gradient_svm(X,W,Y,lambd):
    n = float(Y.shape[0])
    W_1 = W.copy()
    W_1[0] = 0 #for regularization
    Y_cap = np.dot(X,W)
    Y_Y_cap = np.multiply(Y,Y_cap)
    del_Y = Y.copy()
    del_Y[Y_Y_cap>=1]=0    
    grad_temp = np.dot(X.T, del_Y)
    #grad = (lambd*W_1 - grad_temp)/n
    # multipling by n so that : remember I did that mistake so multilpy by n
    grad = (n*lambd*W_1 - grad_temp)/n   
    
    #finding the error
    p = (lambd*np.dot(W_1.T,W_1))/float(2*n)
    zero_mat = np.zeros([Y.shape[0],1])
    J = (1/n)*np.sum(np.maximum(zero_mat,1 - Y_Y_cap)) + p
    
    return grad,J



def support_vector(df_train,df_test):
    X_train = np.array(df_train.drop('class_label_naive',axis=1))
    intercept = np.ones((X_train.shape[0],1))
    X_train = np.hstack((intercept,np.array(X_train)))
    Wt = np.zeros((X_train.shape[1],1))
        
    Y_train = np.array(df_train['class_label_naive'])
    Y_train = Y_train.reshape(Y_train.shape[0],1)
    Y_train[Y_train==0]=-1
    
    num_steps = 100
    err_arr = []
    lambd = 0.01
    alpha = 0.5
    thresh = 10**-6
    for step in range(num_steps):
        grad,err = gradient_svm(X_train,Wt,Y_train,lambd)
        Wt_new = Wt - alpha*grad
        if np.linalg.norm(Wt_new-Wt)<thresh:
            break
        Wt = Wt - alpha*grad
        err_arr.append(err)
        #print(min(err_arr))
    X_test = np.array(df_test.drop('class_label_naive',axis=1))
    intercept = np.ones((X_test.shape[0],1))
    X_test = np.hstack((intercept,np.array(X_test)))
    
    Y_test = np.array(df_test['class_label_naive'])
    Y_test = Y_test.reshape(Y_test.shape[0],1)
    Y_test[Y_test==0]=-1
    
    Y_cap = np.dot(X_test,Wt)
    Y_cap[Y_cap >=0] = 1
    Y_cap[Y_cap <0] = -1
    #print(np.sum(Y_cap == Y_test)/float(Y_cap.shape[0]))
    #return err_arr
    return(error(Y_test, Y_cap))
    #return(np.sum(Y_cap == Y_test)/float(Y_cap.shape[0]))
    
#%%
def q1(df_train_csv_name,df_test_csv_name,lr_svm):
    
    function_to_apply = 1
    w = 4000
    num_feat = 2
    df_train = pd.read_csv(df_train_csv_name, delimiter='\t|\n', header=None, names=['id', 'class_label_naive', 'reviewText'],
                           engine='python')
    df_test = pd.read_csv(df_test_csv_name, delimiter='\t|\n', header=None, names=['id', 'class_label_naive', 'reviewText'],
                          engine='python')


    df_train.reviewText = df_train.reviewText.apply(fnc)
    df_temp_wrds_slt = df_train.copy()

    df_train.reviewText = df_train.reviewText.apply(fnc_vectorize(function_to_apply))
    df_temp_wrds_slt.reviewText = df_temp_wrds_slt.reviewText.apply(fnc_vectorize(1))

    df_test.reviewText = df_test.reviewText.apply(fnc)
    df_test.reviewText = df_test.reviewText.apply(fnc_vectorize(function_to_apply))

    
    wrds_slt = frequent_words(df_temp_wrds_slt, w)
    df_feat_train = bag_of_feature(df_train, wrds_slt,add_class = True)
    df_feat_test = bag_of_feature(df_test, wrds_slt,add_class = True)
    
    if (lr_svm == '1'):
        err = logistics_regression(df_feat_train,df_feat_test)
        print('ZERO-ONE-LOSS-LR {}'.format(err))
    elif(lr_svm == '2'):
        err = support_vector(df_feat_train,df_feat_test)
        print('ZERO-ONE-LOSS-SVM {}'.format(err))
    elif(lr_svm == '3'):
        err = nbc_combined(df_feat_train,num_feat,df_feat_test,wrds_slt)
        print('ZERO-ONE-LOSS-NBC {}'.format(err))
    return 0



#%% Code for analysis

def crossvalidate(yelp_data_csv_filename):
    df = pd.read_csv(yelp_data_csv_filename, delimiter='\t|\n', header=None, names=['id', 'class_label_naive', 'reviewText'],
                           engine='python')
    df = df.iloc[np.random.permutation(len(df))].reset_index(drop=True)
    lst_df = np.array_split(df,10)
    
    #train_size = np.multiply(2000,np.array([0.01, 0.03, 0.05, 0.08, 0.1, 0.15]))
    data_sz = [0.01, 0.03, 0.05, 0.08, 0.1, 0.15]
    train_size = np.multiply(2000,np.array(data_sz))
    #train_size = np.multiply(2000,np.array([0.01]))
    w = 4000
    num_feat = 2
    function_to_apply = 1
    err_svm_arr = []
    err_lr_arr = []
    err_nbc_arr = []
    for tr_sz in train_size:
        err_svm=[]
        err_lr = []
        err_nbc = []
        for i in range(10):
            train_df_mega = pd.concat(lst_df[:i]+lst_df[i+1:]).reset_index(drop=True)
            df_test = lst_df[i].reset_index(drop=True)
            df_train = train_df_mega.sample(tr_sz).reset_index(drop=True)
            
            
            df_train.reviewText = df_train.reviewText.apply(fnc)
            df_temp_wrds_slt = df_train.copy()
            
            
            df_train.reviewText = df_train.reviewText.apply(fnc_vectorize(function_to_apply))
            df_temp_wrds_slt.reviewText = df_temp_wrds_slt.reviewText.apply(fnc_vectorize(1))
            
            
            df_test.reviewText = df_test.reviewText.apply(fnc)
            df_test.reviewText = df_test.reviewText.apply(fnc_vectorize(function_to_apply))
            
            wrds_slt = frequent_words(df_temp_wrds_slt, w)
            df_feat_train = bag_of_feature(df_train, wrds_slt,add_class = True)
            df_feat_test = bag_of_feature(df_test, wrds_slt,add_class = True)   
            
            
            err_svm.append(support_vector(df_feat_train,df_feat_test))
            err_lr.append(logistics_regression(df_feat_train,df_feat_test))
            err_nbc.append(nbc_combined(df_feat_train,num_feat,df_feat_test,wrds_slt))
            #print(df_train.shape[0])
        #print(np.average(err_svm))
        #print(np.average(err_lr))
        #print(np.average(err_nbc))
        sqrt_sz = np.sqrt(10)
        err_svm_arr.append((np.average(err_svm),np.std(err_svm)/sqrt_sz))
        err_lr_arr.append((np.average(err_lr),np.std(err_lr)/sqrt_sz))
        err_nbc_arr.append((np.average(err_nbc),np.std(err_nbc)/sqrt_sz))
    print('(Mean Error,Standard Error): SVM')
    for tup in err_svm_arr:
        print(tup)
    print('(Mean Error,Standard Error): LR')
    for tup in err_lr_arr:
        print(tup)
    print('(Mean Error,Standard Error): NBC')
    for tup in err_nbc_arr:
        print(tup)
    
    
#    import matplotlib.pyplot as plt
#    errorplots = plt.figure()
#    svm_plot = plt.errorbar(x=data_sz,  y=[x[0] for x in err_svm_arr],
#                 yerr=[x[1] for x in err_svm_arr])
#    Lr_plot = plt.errorbar(x=data_sz,  y=[x[0] for x in err_lr_arr],
#                 yerr=[x[1] for x in err_lr_arr])
#    nbc_plot = plt.errorbar(x=data_sz,  y=[x[0] for x in err_nbc_arr],
#                 yerr=[x[1] for x in err_nbc_arr])
#    plt.title("Learning curves and error bars")
#    plt.legend([svm_plot, Lr_plot,nbc_plot], ['SVM', 'LR','NBC'])
#    plt.xlabel('Training Set size')
#    plt.ylabel('Zero One Loss')
#    
#    errorplots.savefig('Without Feature scale')

    
#%%
    
def main():
    num_imputs = len(sys.argv[1:])
    inputs = sys.argv[1:]
    q1(inputs[0],inputs[1],inputs[2])
 
if __name__ == "__main__":
    main()

    
