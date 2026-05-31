from pickle import FALSE
from sklearn.metrics import confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import SimpleITK as sitk
import pandas as pd
import numpy as np
import random
import tensorflow as tf
from numpy.random import seed
seed(1)
tf.random.set_seed(1)
random.seed(1)
import argparse
import os
os.environ['TF_DETERMINISTIC_OPS'] = '1'
import csv
import argparse
import sys


#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-x_test', type=str, help='X test set')
parser.add_argument('-y_test', type=str, help='y test set')
parser.add_argument('-fold', type=str, help='fold split')
parser.add_argument('-path', type=str, help='path to save')
#parser.add_argument('-wl', type=int, help='wavelenght')
args = parser.parse_args()

def loadArrayPatches(df, fold, channels):
  df=df[df['ID'].isin(fold['ID'])]
  ids=df['ID'].unique()
  print(len(ids))
  data_array=np.zeros((len(ids),341,86,channels))
  y_array=np.zeros((len(ids)))
  id_array=0
  for i in ids:
    aux=df[df['ID']==i]
    aux=aux.sort_values(['C#X', 'C#Y'], ascending=[True, True])
    y_array[id_array]=aux['cD#classID'].iloc[0]
    min_row=np.min(aux['C#X'].unique())
    max_row=np.max(aux['C#X'].unique())
    min_col=np.min(aux['C#Y'].unique())
    max_col=np.max(aux['C#Y'].unique())
    r=0
    for r1 in range(min_row,max_row,3):
      c=0
      for c1 in range(min_col,max_col,3):
        for z in range(channels):
          row=aux[aux['C#X']==r1]
          col=row[row['C#Y']==c1]
          match z:
            case 0:
              intensity=col['C#intensity']
              if(len(intensity)>0):
                data_array[id_array,r,c,z]=intensity
            case 1:
              wl_400=col['C#400']
              if(len(wl_400)>0):
                data_array[id_array,r,c,z]=wl_400
            case 2:
              wl_410=col['C#410']
              if(len(wl_410)>0):
                data_array[id_array,r,c,z]=wl_410
            case 3:
              wl_420=col['C#420']
              if(len(wl_420)>0):
                data_array[id_array,r,c,z]=wl_420
            case 4:
              wl_430=col['C#430']
              if(len(wl_430)>0):
                data_array[id_array,r,c,z]=wl_430
            case 5:
              wl_440=col['C#440']
              if(len(wl_440)>0):
                data_array[id_array,r,c,z]=wl_440
            case 6:
              wl_450=col['C#450']
              if(len(wl_450)>0):
                data_array[id_array,r,c,z]=wl_450
            case 7:
              wl_460=col['C#460']
              if(len(wl_460)>0):
                data_array[id_array,r,c,z]=wl_460
            case 8:
              wl_470=col['C#470']
              if(len(wl_470)>0):
                data_array[id_array,r,c,z]=wl_470
            case 9:
              wl_480=col['C#480']
              if(len(wl_480)>0):
                data_array[id_array,r,c,z]=wl_480
            case 10:
              wl_490=col['C#490']
              if(len(wl_490)>0):
                data_array[id_array,r,c,z]=wl_490
            case 11:
              wl_500=col['C#500']
              if(len(wl_500)>0):
                data_array[id_array,r,c,z]=wl_500
            case 12:
              wl_510=col['C#510']
              if(len(wl_510)>0):
                data_array[id_array,r,c,z]=wl_510
            case 13:
              wl_520=col['C#520']
              if(len(wl_520)>0):
                data_array[id_array,r,c,z]=wl_520
            case 14:
              wl_530=col['C#530']
              if(len(wl_530)>0):
                data_array[id_array,r,c,z]=wl_530
            case 15:
              wl_540=col['C#540']
              if(len(wl_540)>0):
                data_array[id_array,r,c,z]=wl_540
            case 16:
              wl_549=col['C#549']
              if(len(wl_549)>0):
                data_array[id_array,r,c,z]=wl_549
            case 17:
              wl_559=col['C#559']
              if(len(wl_559)>0):
                data_array[id_array,r,c,z]=wl_559
            case 18:
              wl_569=col['C#569']
              if(len(wl_569)>0):
                data_array[id_array,r,c,z]=wl_569
            case 19:
              wl_579=col['C#579']
              if(len(wl_579)>0):
                data_array[id_array,r,c,z]=wl_579
            case 20:
              wl_589=col['C#589']
              if(len(wl_589)>0):
                data_array[id_array,r,c,z]=wl_589
            case 21:
              wl_599=col['C#599']
              if(len(wl_599)>0):
                data_array[id_array,r,c,z]=wl_599
            case 22:
              wl_609=col['C#609']
              if(len(wl_609)>0):
                data_array[id_array,r,c,z]=wl_609
            case 23:
              wl_619=col['C#619']
              if(len(wl_619)>0):
                data_array[id_array,r,c,z]=wl_619
            case 24:
              wl_629=col['C#629']
              if(len(wl_629)>0):
                data_array[id_array,r,c,z]=wl_629
            case 25:
              wl_639=col['C#639']
              if(len(wl_639)>0):
                data_array[id_array,r,c,z]=wl_639
            case 26:
              wl_649=col['C#649']
              if(len(wl_649)>0):
                data_array[id_array,r,c,z]=wl_649
            case 27:
              wl_659=col['C#659']
              if(len(wl_659)>0):
                data_array[id_array,r,c,z]=wl_659
            case 28:
              wl_669=col['C#669']
              if(len(wl_669)>0):
                data_array[id_array,r,c,z]=wl_669
            case 29:
              wl_679=col['C#679']
              if(len(wl_679)>0):
                data_array[id_array,r,c,z]=wl_679
            case 30:
              wl_689=col['C#689']
              if(len(wl_689)>0):
                data_array[id_array,r,c,z]=wl_689
            case 31:
              wl_699=col['C#699']
              if(len(wl_699)>0):
                data_array[id_array,r,c,z]=wl_699
            case 32:
              wl_709=col['C#709']
              if(len(wl_709)>0):
                data_array[id_array,r,c,z]=wl_709
        c+=1
      r+=1
    id_array+=1
  return data_array, y_array



def model_eval(y_test, y_pred_raw):
    y_pred = (y_pred_raw>=0.5)
    y_pred = y_pred.astype(int)
    df = pd.DataFrame(y_test)
    #y_test = y_test.to_frame()
    df = df.rename(columns={'cD#classID': 'ground_truth'})
    df['preds'] = y_pred
    df['preds_raw'] = y_pred_raw
    return df

    


fold = args.fold
test = pd.read_csv(fold, low_memory=False)


x_test = np.load(args.x_test)
y_test = np.load(args.y_test)

path=args.path
model=tf.keras.models.load_model(path+".h5")
model.trainable=False



    #Make predictions and save the results
y_pred=model.predict(x_test)
preds = model_eval(y_test, y_pred)

df = pd.merge(preds, test, left_index=True, right_index=True)
df.to_csv(path+'_predictions.csv')





