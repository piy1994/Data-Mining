import numpy as np
import pandas as pd
from pre_proc import error

class decision_tree_boost:
    def __init__(self,max_depth,min_size,classes = [-1,1]):
        self.max_depth = max_depth
        self.min_size = min_size
        self.class_labels = classes
        self.root = None
        
    def sum_pos(self,x):
        return sum([num for num in x if num>=0 ])
    def sum_neg(self,x):
        return sum([num for num in x if num<0 ])
    
    def gini_index(self,split_grps, classes):#split groups y label value it means that
        gini_index_grp = 0.0
        total_size = 0.0
        for grp in split_grps:
            size = float(sum(map(abs,grp)))
            total_size+=size
            if size == 0:
                continue
            prob_sq = (self.sum_pos(grp)/size)**2 + (self.sum_neg(grp)/size)**2
            gini_index_grp += size*(1 - prob_sq)
        return gini_index_grp/total_size
        
    def dataset_split(self,index,dataset):#index means the feature number 
        left, right = list(), list()
        for row in dataset:
            if row[index] == 0:
                left.append(row)
            else:
                right.append(row)
        return left,right
    
    def y_label(self,x):
        return [row[-1] for row in x]
        
    
    def get_best_split(self,dataset,classes = [-1,1],max_feature_size = 40000):
        best_index, best_score, best_groups = max_feature_size,999, None
        for index in range(len(dataset[0])-1):
            group_left,group_right = self.dataset_split(index,dataset)
            gini = self.gini_index([self.y_label(group_left),self.y_label(group_right)], classes) #
            if gini < best_score:
                best_index,best_score,best_groups = index, gini, [group_left,group_right]
        #if (best_index == 40000):
            #print(dataset)
            #np.save('error',dataset)
        return {'index':best_index, 'score':best_score, 'groups':best_groups}
 ####################       
    def leaf_node(self,group):
        predictions = sum([row[-1] for row in group])
        if predictions >=0:
            return 1.0
        else:
            return -1.0
######################    
    def recursive_splitting(self,node,max_depth,min_size,curr_depth):
        #print(curr_depth)
        try:
            left,right = node['groups']
        except:
            print(node)
        del(node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.leaf_node(left + right)
            return
        if curr_depth>=max_depth:
            node['left'],node['right']=self.leaf_node(left),self.leaf_node(right)
            return
        
        if (len(left)<min_size):
            node['left'] = self.leaf_node(left)
        elif np.sum(np.array(left)[:,-1] <0) == len(left):
            node['left'] = self.leaf_node(left)
        elif np.sum(np.array(left)[:,-1] >=0) == len(left):
            node['left'] = self.leaf_node(left)
        else:
            node['left'] = self.get_best_split(left)
            self.recursive_splitting(node['left'],max_depth,min_size,curr_depth+1)
        
        if (len(right)<min_size):
            node['right'] = self.leaf_node(right)
        elif np.sum(np.array(right)[:,-1] <0) == len(right):
            node['right'] = self.leaf_node(right)
        elif np.sum(np.array(right)[:,-1] >=0) == len(right):
            node['right'] = self.leaf_node(right)
        else:
            node['right'] = self.get_best_split(right)
            self.recursive_splitting(node['right'],max_depth,min_size,curr_depth+1)
            
    def train(self,train_dataset):
        self.root = self.get_best_split(train_dataset)
        self.recursive_splitting(self.root,self.max_depth,self.min_size,1)
        return
        
    def predict_row(self,node, row):
        if row[node['index']] == 0:
            if isinstance(node['left'], dict):
                return self.predict_row(node['left'], row)
            else:
                return node['left']
        else:
            if isinstance(node['right'], dict):
                return self.predict_row(node['right'], row)
            else:
                return node['right']
    
    def predict(self,X_test):
        predict_y_test = []
        for row in X_test:
            predict_y_test.append(self.predict_row(self.root,row))
        return predict_y_test
        

class boosting:
#we convert to list the array numpy
    def __init__(self,max_depth,min_size,num_trees):
        self.max_depth = max_depth
        self.min_size = min_size
        self.num_trees = num_trees
        self.lst_dt = None
        self.alphas = []
    def train_boost(self,X,Y):
        ep = 10**-6
        et_arr = []
        lst_dt = []
        #sz = int(dataset.shape[0]*self.sz_pro)
        wt = np.ones(shape = Y.shape)
        wt = wt/np.sum(wt)
        wt_y = np.multiply(wt,Y)
        dataset =  np.hstack((X,wt_y))
        for i in range(self.num_trees):
            print('on {} tree'.format(i+1))
            d_t = decision_tree_boost(max_depth=self.max_depth,min_size=self.min_size)
            d_t.train(dataset.tolist())
            lst_dt.append(d_t)
            
            y_predict = d_t.predict(np.delete(dataset,-1,axis=1))
            y_predict = np.array(y_predict).reshape(Y.shape)
            
            et = np.sum(wt[Y !=y_predict])
            et_arr.append(et)
            if et == 0:
                alpha = (0.5)*np.log(((1 - et + ep)/(et + ep)))
                self.alphas.append(alpha)
                break
            Y_Y_pred = np.multiply(Y,y_predict)
            alpha = 0.5*np.log((1-et)/et)
            self.alphas.append(alpha)
            
            wt = np.multiply(wt,np.exp(-1.0*alpha*Y_Y_pred))
            #print(np.sum(wt))
            wt = wt/np.sum(wt)
           # print(np.sum(wt))
            
            reshape_shape = dataset[:,-1].shape
            dataset[:,-1] = np.multiply(wt,Y).reshape(reshape_shape)         
            
        self.lst_dt = lst_dt
        return et_arr
        
    def test_boost(self,X):
        pred_y_arr = []
        i=0
        for dt in self.lst_dt:
            pred_y = dt.predict(X.tolist())
            #print(max(pred_y))
            #print(min(pred_y))
            pred_y_arr.append(self.alphas[i]*np.array(pred_y))
            i+=1
            #print('testing {}'.format(i))
        return pred_y_arr

    
def boosting_function(X_train,Y_train,X_test,Y_test,max_depth,min_size,num_trees):
    Y_train[Y_train==0] = -1.0
    Y_train[Y_train==1] = 1.0
    Y_test[Y_test==0] = -1.0
    Y_test[Y_test==1] = 1.0   
    
    Y_train = Y_train.reshape((X_train.shape[0],1))
    Y_test = Y_test.reshape((X_test.shape[0],1))
    
    boost_t= boosting(max_depth=max_depth,min_size=min_size,num_trees=num_trees)
    et_arr = boost_t.train_boost(X_train,Y_train)

    pred_y_arr = np.array(boost_t.test_boost(X_test))

    y_test_predict = np.sum(pred_y_arr,axis=0)
    y_test_predict[y_test_predict>=0]=1
    y_test_predict[y_test_predict<0]=-1
    err = error(list(Y_test[:,0]),list(y_test_predict))
    return err
    
    
    

