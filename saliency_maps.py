
import numpy as np
import pandas as pd
import nibabel as nib
import nibabel.processing
import SimpleITK as sitk
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
import tf_keras_vis
from tensorflow.keras import backend as K
from tf_keras_vis.saliency import Saliency
from tf_keras_vis.utils.scores import CategoricalScore
from tf_keras_vis.utils.scores import BinaryScore
from matplotlib import cm
import os
import pickle
import pandas as pd

import argparse
import sys


#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str, help='path')
args = parser.parse_args()

# Function to plot and save the image and its saliency map
def plot_and_save_image_and_saliency(img_array, saliency, output_path):
    plt.figure(figsize=(8, 8))

    plt.subplot(1, 2, 1)
    plt.imshow(img_array[0].astype(np.uint8))
    plt.title('Original Image')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(saliency, cmap='hot')
    plt.title('Saliency Map')
    plt.axis('off')

    plt.savefig(output_path)
    plt.show()

def model_modifier_function(cloned_model):
    cloned_model.layers[-1].activation = tf.keras.activations.linear



#------------------- parameters and directory paths ----------------------------#
#N_MAPS = 2 #number of subjects of each TP/TN/FP/FN array to generate saliency maps for
SMOOTH_SAMPLES = 20 #number of smoothgrad iterations
SMOOTH_NOISE = 0.2 #smoothgrad gaussian noise spread
#------------------- ------------------------------ ----------------------------#
path=args.path
for fold in range (0,10,1):

    #load model to generate saliency maps for
    model = tf.keras.models.load_model(path+"best_model_fold"+str(fold)+"_best.h5")
    model.summary()

    #load image data 
    array = np.load(path+"x_val_"+str(fold)+".npy")


    #score function for saliency (ensures the saliency map is calculated wrt the correct class)
    #score_function = CategoricalScore([2])
    score_function = BinaryScore(True)


    # Create Saliency object.
    saliency = Saliency(model,
                        model_modifier=model_modifier_function,
                        clone=True)

    saliency_data = []



    for i in range(len(array)):
        print('generating saliency map for {}'.format(i))
        aux=array[i,:,:]
        print(aux.shape)
        aux_input = np.expand_dims(aux, axis=(0, -1))
        saliency_map = saliency(score_function,
                            aux_input,
                            smooth_samples=SMOOTH_SAMPLES, # The number of calculating gradients iterations.
                            smooth_noise=SMOOTH_NOISE) # noise spread level.
        np.save(path+'saliency_maps_fold_'+str(fold)+'/saliency_map_fold_'+str(fold)+'_'+str(i)+'.npy', saliency_map)


    

