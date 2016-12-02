# -*- coding: utf-8 -*-
"""
Created on Thu Nov 24 10:10:17 2016

@author: revan
"""

# A bit of a setup

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st

from cs231n.classifiers.two_layer_neural_net import TwoLayerNet
from cs231n.classifiers.neural_net import ThreeLayerNet
from cs231n.data_utils import load_CIFAR10

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))
  
def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    cifar10_dir = 'cs231n/datasets/cifar-10-batches-py'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)
        
    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]
    
    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    # Reshape data to rows
    X_train = X_train.reshape(num_training, -1)
    X_val = X_val.reshape(num_validation, -1)
    X_test = X_test.reshape(num_test, -1)

    return X_train, y_train, X_val, y_val, X_test, y_test
    
X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data()

print 'Train data shape: ', X_train.shape
print 'Train labels shape: ', y_train.shape
print 'Validation data shape: ', X_val.shape
print 'Validation labels shape: ', y_val.shape
print 'Test data shape: ', X_test.shape
print 'Test labels shape: ', y_test.shape 

best_val = -1
best_net = None
best_stats = None
best_nets_ensamble = []
values_ensamble = []
best_stats_ensamble = []
results = {} 


num_classes = 10
input_size = 32 * 32 * 3

for i in xrange(1):   
    neurons = np.random.randint(100, 1000)
    lr = np.random.uniform(1e-4, 1e-6)
    rs = np.random.uniform(0, 0.5)
    iters = np.random.randint(2000, 5000)
    learning_rate_decay = np.random.uniform(0.9, 1)
    p = np.random.uniform(0.2, 0.6)
    opt = np.random.choice(['Nesterov', 'Adagrad', 'RMSProp', 'Adam'])
    
    net = np.random.choice([TwoLayerNet(input_size, neurons, num_classes), ThreeLayerNet(input_size, neurons, num_classes)])

    # Train the network
    stats = net.train(X_train, y_train, X_val, y_val, lr, learning_rate_decay,
            rs, iters, opt, p, 256, False, False)

    y_train_pred = net.predict(X_train)
    acc_train = np.mean(y_train == y_train_pred)
    y_val_pred = net.predict(X_val)
    acc_val = np.mean(y_val == y_val_pred)
    
    num_layers = 0
    if type(net) == TwoLayerNet:
        num_layers = 2
    else:
        num_layers = 3
        
    optimizers = {'Nesterov': 1, 'Adagrad': 2, 'RMSProp': 3, 'Adam': 4}
    
    results[(lr, rs, neurons, learning_rate_decay, iters, num_layers, optimizers[opt], p)] = (acc_train, acc_val)
    
    accepted_accuracy = 0.52
    
    if acc_val > accepted_accuracy:
        best_nets_ensamble.append(net)
        values_ensamble.append(acc_val)
        best_stats_ensamble.append(stats)
        
    number_of_ensemblers = len(best_nets_ensamble)    

    if best_val < acc_val:
        best_stats = stats
        best_val = acc_val
        best_net = net
        
    print values_ensamble        
    
# Print out results.
for lr, reg, neurons, learning_rate_decay, iters, num_layers, opts, p  in sorted(results):
    train_accuracy, val_accuracy = results[(lr, reg, neurons, learning_rate_decay, iters, num_layers, opts, p)]
    print 'lr %e reg %e neur%e lrd %e iters %e num_layers %e opt %e p %e train accuracy: %f val accuracy: %f' % (
                lr, reg, neurons, learning_rate_decay, iters, num_layers, opts, p, train_accuracy, val_accuracy)
    
print 'best validation accuracy achieved during cross-validation: %f' % best_val  

test_acc = (best_net.predict(X_test) == y_test).mean()
print number_of_ensemblers

if number_of_ensemblers > 0:
    ensemble_net = np.zeros((number_of_ensemblers, len(y_test)))
    i = 0
    for net in best_nets_ensamble:
        pred = net.predict(X_test)
        ensemble_net[i,:] = pred
        i += 1
    
    pred_final = st.mode(ensemble_net)   
    pred_final = pred_final[0]

    accuracy_ensembler = (pred_final == y_test).mean()
    
else:
    accuracy_ensembler = test_acc    

print 'Test accuracy - best model: ', test_acc
print 'Test accuracy - ensembler: ', accuracy_ensembler

o = 'results_softmax_CIFAR.txt'
output = open(o, "wb")
output.write('Test accuracy - best model: ' + str(test_acc) + '\n')
output.write('Test accuracy - ensembler model: ' + str(accuracy_ensembler) + '\n')  
output.close()