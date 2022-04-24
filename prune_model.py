#================================================================
#
#   File name   : prune_model.py
#   Author      : Aleksei Zubkov
#   Created date: 2022-04-10
#   Description : used to prune the model and save the result
#
#================================================================

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.python.saved_model import tag_constants
from yolov3.dataset import Dataset
from yolov3.yolov4 import Create_Yolo
from yolov3.utils import load_yolo_weights, detect_image, image_preprocess, postprocess_boxes, nms, read_class_names
from yolov3.configs import *
from numpy import linalg as LA
import shutil
import json
import time


def weight_prune_dense_layer(k_weights, b_weights, k_sparsity):
    """
    Takes in matrices of kernel and bias weights (for a dense
      layer) and returns the unit-pruned versions of each
    Args:
      k_weights: 2D matrix of the 
      b_weights: 1D matrix of the biases of a dense layer
      k_sparsity: percentage of weights to set to 0
    Returns:
      kernel_weights: sparse matrix with same shape as the original
        kernel weight matrix
      bias_weights: sparse array with same shape as the original
        bias array
    """
    # Copy the kernel weights and get ranked indeces of the abs
    kernel_weights = np.copy(k_weights)
    ind = np.unravel_index(
        np.argsort(
            np.abs(kernel_weights),
            axis=None),
        kernel_weights.shape)
        
    # Number of indexes to set to 0
    cutoff = int(len(ind[0])*k_sparsity)
    # The indexes in the 2D kernel weight matrix to set to 0
    sparse_cutoff_inds = (ind[0][0:cutoff], ind[1][0:cutoff])
    kernel_weights[sparse_cutoff_inds] = 0.
        
    # Copy the bias weights and get ranked indeces of the abs
    bias_weights = np.copy(b_weights)
    ind = np.unravel_index(
        np.argsort(
            np.abs(bias_weights), 
            axis=None), 
        bias_weights.shape)
        
    # Number of indexes to set to 0
    cutoff = int(len(ind[0])*k_sparsity)
    # The indexes in the 1D bias weight matrix to set to 0
    sparse_cutoff_inds = (ind[0][0:cutoff])
    bias_weights[sparse_cutoff_inds] = 0.
    
    return kernel_weights, bias_weights



def unit_prune_dense_layer(k_weights, b_weights, k_sparsity):
    """
    Takes in matrices of kernel and bias weights (for a dense
      layer) and returns the unit-pruned versions of each
    Args:
      k_weights: 2D matrix of the 
      b_weights: 1D matrix of the biases of a dense layer
      k_sparsity: percentage of weights to set to 0
    Returns:
      kernel_weights: sparse matrix with same shape as the original
        kernel weight matrix
      bias_weights: sparse array with same shape as the original
        bias array
    """

    # Copy the kernel weights and get ranked indeces of the
    # column-wise L2 Norms
    kernel_weights = np.copy(k_weights)
    ind = np.argsort(LA.norm(kernel_weights, axis=0))
        
    # Number of indexes to set to 0
    cutoff = int(len(ind)*k_sparsity)
    # The indexes in the 2D kernel weight matrix to set to 0
    sparse_cutoff_inds = ind[0:cutoff]
    kernel_weights[:,sparse_cutoff_inds] = 0.
        
    # Copy the bias weights and get ranked indeces of the abs
    bias_weights = np.copy(b_weights)
    # The indexes in the 1D bias weight matrix to set to 0
    # Equal to the indexes of the columns that were removed in this case
    #sparse_cutoff_inds
    bias_weights[sparse_cutoff_inds] = 0.
    
    return kernel_weights, bias_weights



def sparsify_model(model, k_sparsity, pruning='weight'):
    """
    Takes in a model made of dense layers and prunes the weights
    Args:
      model: Keras model
      k_sparsity: target sparsity of the model
    Returns:
      sparse_model: sparsified copy of the previous model
    """
    # Copying a temporary sparse model from our original
    # sparse_model = tf.keras.models.clone_model(model)
    # sparse_model.set_weights(model.get_weights())

    sparse_model=model
    
    # Getting a list of the names of each component (w + b) of each layer
    names = [weight.name for layer in sparse_model.layers for weight in layer.weights]
    print(names)
    # Getting the list of the weights for each component (w + b) of each layer
    weights = sparse_model.get_weights()
    
    # Initializing list that will contain the new sparse weights
    newWeightList = []

    # Iterate over all but the final 2 layers (the softmax)
    for i in range(0, len(weights)-2, 2):
        
        if pruning=='weight':
            kernel_weights, bias_weights = weight_prune_dense_layer(weights[i],
                                                                    weights[i+1],
                                                                    k_sparsity)
        elif pruning=='unit':
            kernel_weights, bias_weights = unit_prune_dense_layer(weights[i],
                                                                  weights[i+1],
                                                                  k_sparsity)
        else:
            print('does not match available pruning methods ( weight | unit )')
        
        # Append the new weight list with our sparsified kernel weights
        newWeightList.append(kernel_weights)
        
        # Append the new weight list with our sparsified bias weights
        newWeightList.append(bias_weights)

    # Adding the unchanged weights of the final 2 layers
    for i in range(len(weights)-2, len(weights)):
        unmodified_weight = np.copy(weights[i])
        newWeightList.append(unmodified_weight)

    # Setting the weights of our model to the new ones
    sparse_model.set_weights(newWeightList)
    
    # Re-compiling the Keras model (necessary for using `evaluate()`)
    sparse_model.compile(
        loss=tf.keras.losses.categorical_crossentropy,
        optimizer='adam',
        metrics=['accuracy'])
    
    # Printing the the associated loss & Accuracy for the k% sparsity
    # score = sparse_model.evaluate(x_test, y_test, verbose=0)
    # print('k% weight sparsity: ', k_sparsity,
    #       '\tTest loss: {:07.5f}'.format(score[0]),
    #       '\tTest accuracy: {:05.2f} %%'.format(score[1]*100.))
    
    return sparse_model, score


if __name__ == '__main__':       

    # list of sparsities
    k_sparsities = [0.0, 0.25, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.97, 0.99]

    # The empty lists where we will store our training results
    mnist_model_loss_weight = []
    mnist_model_accs_weight = []
    mnist_model_loss_unit = []
    mnist_model_accs_unit = []

    Darknet_weights = YOLO_V3_TINY_WEIGHTS
    yolo = Create_Yolo(input_size=YOLO_INPUT_SIZE, CLASSES=YOLO_COCO_CLASSES)
    #load_yolo_weights(yolo, Darknet_weights) # use Darknet weights
    # yolo.compile(
    #     loss=tf.keras.losses.categorical_crossentropy,
    #     optimizer='adam',
    #     metrics=['accuracy'])
    

    dataset = 'mnist'
    pruning = 'weight'
    print('\n MNIST Weight-pruning\n')
    for k_sparsity in k_sparsities:
        sparse_model, score = sparsify_model(yolo, 
                                            k_sparsity=k_sparsity, 
                                            pruning=pruning)
        mnist_model_loss_weight.append(score[0])
        mnist_model_accs_weight.append(score[1])
        
        # Save entire model to an H5 file
        sparse_model.save('pruned_models/sparse-model_k-{}_{}-pruned.h5'.format(k_sparsity, pruning))
        del sparse_model


    # pruning='unit'
    # print('\n MNIST Unit-pruning\n')
    # for k_sparsity in k_sparsities:
    #     sparse_model, score = sparsify_model(yolo,  
    #                                         k_sparsity=k_sparsity, 
    #                                         pruning=pruning)
    #     mnist_model_loss_unit.append(score[0])
    #     mnist_model_accs_unit.append(score[1])
        
    #     # Save entire model to an H5 file
    #     sparse_model.save('pruned_models/sparse_{}-model_k-{}_{}-pruned.h5'.format(dataset, k_sparsity, pruning))
    #     del sparse_model