import numpy as np
import argparse
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize,RegexpTokenizer
from nltk.util import ngrams
import random
from nltk.stem.snowball import SnowballStemmer
from datetime import datetime
import sys


def accuracy_predict(y_p,y_act):
    return float(np.sum(y_p == y_act))/y_p.shape[0]

def make_dictionary(sentences):
    stemmer = SnowballStemmer("english")
    stopWords = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer(r'\w+')
    word_dict = {}
    for each_sentence in sentences:
        tokens = tokenizer.tokenize(unicode(each_sentence.lower(), errors='ignore'))
        filtered_words = [stemmer.stem(w) for w in tokens if not w in stopWords]
        dict_update(word_dict,filtered_words)
    return word_dict
        
def dict_update(word_dict,filtered_words):
    for word in filtered_words:
        if word in word_dict:
            word_dict[word]+=1
        else:
            word_dict[word]=1

def make_features(sentences,word_keys):
    stemmer = SnowballStemmer("english")
    num_feat = len(word_keys)
    X_feat = np.zeros(shape = (len(sentences),num_feat))
    i = 0
    stopWords = set(stopwords.words("english"))
    tokenizer = RegexpTokenizer(r'\w+')
    for each_sentence in sentences:
        this_feat = []
        tokens = tokenizer.tokenize(unicode(each_sentence.lower(), errors='ignore'))
        filtered_words = [stemmer.stem(w) for w in tokens if not w in stopWords]

        for top_word in word_keys:
            
            if top_word in filtered_words:
                this_feat.append(1)
            else:
                this_feat.append(0)
        X_feat[i,:] = np.reshape(np.array(this_feat),newshape = (1,num_feat))
        i+=1
    return X_feat
    
def make_unigrams_train_test_both(sentences,labels,train_size_ratio):
    
    sen_lab = zip(sentences,labels)
    random.shuffle(sen_lab)
    sentences,labels = zip(*sen_lab)
    train_size = int(train_size_ratio*len(sentences))
    sentences_train = sentences[:train_size]
    
    y_train = np.reshape(np.array(labels[:train_size]),newshape = (len(labels[:train_size]),1))
    sentences_test = sentences[train_size:]
    y_test = np.reshape(np.array(labels[train_size:]),newshape = (len(labels[train_size:]),1))
    
    
    
    word_dict = make_dictionary(sentences_train)
    key_val = zip(word_dict.keys(), word_dict.values())
    key_val.sort(key=lambda x: x[1], reverse=True)
        
    top_k = 3000
    word_keys = [val[0] for val in key_val][:top_k]
    
    X_feat_train = make_features(sentences_train,word_keys)
    X_feat_test = make_features(sentences_test,word_keys)
    return X_feat_train,X_feat_test,y_train,y_test,word_keys,top_k,key_val


class Classifier(object):
    def __init__(self):
        pass

    def train():
        """
        Override this method in your class to implement train
        """
        raise NotImplementedError("Train method not implemented")

    def inference():
        """
        Override this method in your class to implement inference
        """
        raise NotImplementedError("Inference method not implemented")

class MLP(Classifier):
    def __init__(self,num_hidden_layer_neuron,num_epoch,learning_rate,lamda):
        super(MLP, self).__init__()
        self.num_hidden_neuron = num_hidden_layer_neuron
        self.num_epoch  = num_epoch
        self.lr = learning_rate
        self.lamda = lamda
        self.wh = None
        self.wout = None
        self.bh = None
        self.bout = None
        #wh,bh,wout,bout,output,E_at_each_iteration = self.training(X_input,y_input)
        

    def sigmoid(self,x,deriv = False):
        if deriv == False:
            return 1/(1 + np.exp(-x))
        else:
            return x*(1-x)
    def train(self,X,y):
        input_layer_neuron = X.shape[1]
        out_layer_neuron = 1
        
        # weight and bias initialization
        wh = 2*np.random.uniform(size = (input_layer_neuron,self.num_hidden_neuron)) - 1
        bh = 2*np.random.uniform(size = (1,self.num_hidden_neuron))-1
        wout = 2*np.random.uniform(size = (self.num_hidden_neuron,out_layer_neuron))-1
        bout = 2*np.random.uniform(size = (1,out_layer_neuron))-1
        E_at_each_iteration = []
        
        for i in range(self.num_epoch):
            # Forward Propogation
            hidden_layer_input = np.dot(X,wh) + bh
            hidden_layer_activation = self.sigmoid(hidden_layer_input)
            outer_layer_input = np.dot(hidden_layer_activation,wout) + bout
            output = self.sigmoid(outer_layer_input)
            
            #backpropagation
            E = y - output
            
            #print 'Iteration number {}, Error : {}'.format(i,np.sqrt(np.sum(E**2)))
            E_at_each_iteration.append(E)
            slope_output_layer = self.sigmoid(output,deriv= True)
            d_output_layer = E*slope_output_layer
            
            E_hidden = d_output_layer.dot(wout.T)
            slope_hidden_layer = self.sigmoid(hidden_layer_activation,deriv=True)
            d_hidden_layer = E_hidden*slope_hidden_layer
            
            
            d_wout = hidden_layer_activation.T.dot(d_output_layer)
            d_wh = X.T.dot(d_hidden_layer)
            
            # regularization
            #self.lamda = 0.02
            d_wout -= self.lamda*wout
            d_wh -= self.lamda*wh
            #updating weights
            wout+= d_wout*self.lr
            bout+= np.sum(d_output_layer,axis = 0,keepdims = True)*self.lr
            wh+= d_wh*self.lr
            bh+= np.sum(d_hidden_layer,axis = 0,keepdims = True)*self.lr
        self.wh = wh
        self.bh = bh
        self.wout = wout
        self.bout = bout
        
        return output,E_at_each_iteration
    
    def inference(self,X):
            hidden_layer_input = np.dot(X,self.wh) + self.bh
            hidden_layer_activation = self.sigmoid(hidden_layer_input)
            outer_layer_input = np.dot(hidden_layer_activation,self.wout) + self.bout
            output = self.sigmoid(outer_layer_input)
            return np.asarray(np.array((output)>0.5),float)



class Perceptron(Classifier):
    def __init__(self,learning_rate,epochs):
        super(Perceptron,self).__init__()
        self.lr = learning_rate
        self.epochs = epochs
        self.wt = None
    
    def train(self,X,y):
        X = np.hstack((np.ones(shape = (X.shape[0],1)),X))
        input_layer_neuron = X.shape[1]
        wt = 2*np.random.uniform(size = (input_layer_neuron,1)) - 1
        for i in xrange(self.epochs):
            y_p = np.dot(X,wt)
            error = y - np.asarray(np.array((y_p)>0),float)
            #print np.sqrt(np.sum(error**2))
            wt+= X.T.dot(error)*self.lr

        self.wt = wt
        #print 'Trained on the network'
        return 
    def inference(self,X_test):
        X_test = np.hstack((np.ones(shape = (X_test.shape[0],1)),X_test))
        y_p = np.dot(X_test,self.wt)
        return np.asarray(np.array((y_p)>0),float)



def feature_extractor(data,labels,word_keys=None,making_for_test = False):
    """
    implement your feature extractor here
    """
    if making_for_test == False:
        X_feat_train,X_feat_test,y_train,y_test,word_keys,_,_ = \
        make_unigrams_train_test_both(sentences=data,labels=labels,train_size_ratio=0.8)
        return X_feat_train,X_feat_test,y_train,y_test,word_keys
    else:
        X_feat_test = make_features(data,word_keys)
        return X_feat_test

def evaluate(preds, golds):
    tp, pp, cp = 0.0, 0.0, 0.0
    for pred, gold in zip(preds, golds):
        if pred == 1:
            pp += 1
        if gold == 1:
            cp += 1
        if pred == 1 and gold == 1:
            tp += 1
    precision = tp / pp
    recall = tp / cp
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        return (precision, recall, 0.0)
    return (precision, recall, f1)

def main():

    argparser = argparse.ArgumentParser()
    with open("sentences.txt") as f:
        data = f.readlines()
    with open("labels.txt") as g:
        labels = [int(label) for label in g.read()[:-1].split("\n")]
    
    """ extracting the features !!"""
    X_feat_train,X_feat_test,y_train,y_test,word_keys = feature_extractor(data=data,labels=labels,word_keys=None,making_for_test = False)
    
    mymlp = MLP(num_hidden_layer_neuron=10,num_epoch=1000,learning_rate=0.01,lamda= 0.01)
    myperceptron = Perceptron(learning_rate=0.09,epochs=1000)

    _,_ = mymlp.train(X_feat_train,y_train)
    myperceptron.train(X_feat_train,y_train)
    
    """
    Testing on testing data set
    """

    predicted_y = mymlp.inference(X_feat_test)
    precision, recall, f1 = evaluate(predicted_y, y_test)
    print "MLP results", precision, recall, f1

    predicted_y = myperceptron.inference(X_feat_test)
    precision, recall, f1 = evaluate(predicted_y, y_test)
    print "Perceptron results", precision, recall, f1

    """
    Testing on unseen testing data in grading
    """
    
    print "Real Testing on unseen data"
    argparser.add_argument("--test_data", type=str, default="../test_sentences.txt", help="The real testing data in grading")
    argparser.add_argument("--test_labels", type=str, default="../test_labels.txt", help="The labels for the real testing data in grading")

    parsed_args = argparser.parse_args(sys.argv[1:])
    real_test_sentences = parsed_args.test_data
    real_test_labels = parsed_args.test_labels
    with open(real_test_sentences) as f:
        real_test_x = f.readlines()
    with open(real_test_labels) as g:
        real_test_y = [int(label) for label in g.read()[:-1].split("\n")]
    
    X_feat_test_real = feature_extractor(data=real_test_x,labels=None,word_keys=word_keys,making_for_test = True)
    real_test_y = np.reshape(np.array(real_test_y),newshape = (len(real_test_y),1))
    
    predicted_y = mymlp.inference(X_feat_test_real)
    precision, recall, f1 = evaluate(predicted_y, real_test_y)
    print "MLP results", precision, recall, f1

    predicted_y = myperceptron.inference(X_feat_test_real)
    precision, recall, f1 = evaluate(predicted_y, real_test_y)
    print "Perceptron results", precision, recall, f1
    

    

if __name__ == '__main__':
    startTime = datetime.now()
    main()
    print "Time taken = {}".format(datetime.now() - startTime)
    
