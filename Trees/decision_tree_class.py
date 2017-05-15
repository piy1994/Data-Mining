import numpy as np
import pandas as pd
from pre_proc import error

class decision_tree:
    def __init__(self,max_depth,min_size,classes = [0,1]):
        self.max_depth = max_depth
        self.min_size = min_size
        self.class_labels = classes
        self.root = None
    
    def gini_index(self,split_grps, classes):#split groups y label value it means that
        gini_index_grp = 0.0
        total_size = 0.0
        for grp in split_grps:
            size = float(len(grp))
            total_size+=size
            if size == 0:
                continue
            prob_sq = 0
            for cls in classes:
                prob_sq += (grp.count(cls)/size)**2
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
        
    
    def get_best_split(self,dataset,classes = [0,1],max_feature_size = 40000):
        best_index, best_score, best_groups = max_feature_size,999, None
        for index in range(len(dataset[0])-1):
            group_left,group_right = self.dataset_split(index,dataset)
            gini = self.gini_index([self.y_label(group_left),self.y_label(group_right)], classes) #
            if gini < best_score:
                best_index,best_score,best_groups = index, gini, [group_left,group_right]
        return {'index':best_index, 'score':best_score, 'groups':best_groups}
        
    def leaf_node(self,group):
        predictions = [row[-1] for row in group]
        return max(set(predictions), key=predictions.count)
    
    def recursive_splitting(self,node,max_depth,min_size,curr_depth):
        #print(curr_depth)
        left,right = node['groups']
        del(node['groups'])
        if not left or not right:
            node['left'] = node['right'] = self.leaf_node(left + right)
            return
        if curr_depth>=max_depth:
            node['left'],node['right']=self.leaf_node(left),self.leaf_node(right)
            return
        
        if (len(left)<min_size):
            node['left'] = self.leaf_node(left)
        elif (len(set([rw[-1] for rw in left]))==1):
            node['left'] = self.leaf_node(left)
        else:
            node['left'] = self.get_best_split(left)
            self.recursive_splitting(node['left'],max_depth,min_size,curr_depth+1)
        
        if (len(right)<min_size):
            node['right'] = self.leaf_node(right)
        elif (len(set([rw[-1] for rw in right]))==1):
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
        
       
        
def decision_fnc(XY_train,XY_test,max_depth,min_size):
    d_t= decision_tree(max_depth=max_depth,min_size=min_size)
    d_t.train(XY_train.tolist())
    pred_y = d_t.predict(np.delete(XY_test,-1,axis=1).tolist())
    original_label = list(XY_test[:,-1])
    pred_y = map(int,pred_y)
    err = error(original_label,list(pred_y))
    return err
    
