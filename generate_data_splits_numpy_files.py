import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.utils import shuffle
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew, kurtosis



from sklearn.model_selection import GroupKFold
import numpy as np

def fiveFold(data, path, col="ID_mouse"):

  X=data["mouse_id_sample_id"]
  y=data['cD#classID']
  #y=data['classID']
  mouse_ids=data["ID_mouse"]
  # Set up GroupKFold to create folds by mouse_id
  gkf = GroupKFold(n_splits=10)

  for fold, (train_idx, val_idx) in enumerate(gkf.split(X, y, groups=mouse_ids)):
      print(f"Fold {fold+1}")
      print(f"Train mice: {np.unique(mouse_ids[train_idx])}")
      print(f"Validation mice: {np.unique(mouse_ids[val_idx])}")
      
      X_train, X_val = X[train_idx], X[val_idx]


      train=data[data['mouse_id_sample_id'].isin(X_train)]
      train = train.filter(['mouse_id_sample_id', 'classID'])
      train.drop_duplicates(inplace=True)
      train = train.sample(frac=1).reset_index(drop=True)


      val=data[data['mouse_id_sample_id'].isin(X_val)]
      val = val.filter(['mouse_id_sample_id', 'classID'])
      val.drop_duplicates(inplace=True)
      val = val.sample(frac=1).reset_index(drop=True)

      train.to_csv(path+"/train"+str(fold)+".csv")
      val.to_csv(path+"/val"+str(fold)+".csv")

       

def create2Ddata(df, fold, f, fn, path,insize):
    print("creating array")
    samples = fold['mouse_id_sample_id'].unique()

    print(f"Number of samples: {len(samples)}")
    
    # Initialize data and label arrays
    data_array = np.zeros((len(samples), insize, 32)) #1000 or 100 or 10
    y_array = np.zeros((len(samples),))
    
    # Define the 32 wavelength columns
    wls = ["C#400","C#410","C#420","C#430","C#440", "C#450","C#460","C#470",
           "C#480","C#490", "C#500","C#510","C#520","C#530","C#540","C#549",
           "C#559","C#569", "C#579", "C#589", "C#599", "C#609", "C#619",
           "C#629", "C#639", "C#649", "C#659", "C#669", "C#679", "C#689",
           "C#699", "C#709"]
    
    # wls = ["L400","L410","L420","L430","L440", "L450","L460","L470",
    #        "L480","L490", "L500","L510","L520","L530","L540","L549",
    #        "L559","L569", "L579", "L589", "L599", "L609", "L619",
    #        "L629", "L639", "L649", "L659", "L669", "L679", "L689",
    #        "L699", "L709"]

    for i, s in enumerate(samples[:len(samples)]):  
        aux = df[df['mouse_id_sample_id'] == s]

        if aux.shape[0] != insize: #1000 or 100 or 10
            raise ValueError(f"Sample {s} has {aux.shape[0]} rows.")

        # Store class label (first value)
        y_array[i] = aux['cD#classID'].iloc[0]

        # Store 1000x32 feature matrix
        data_array[i] = aux[wls].to_numpy()

    print(f"Data array shape: {data_array.shape}")  # Expected: (#samples, 1000 or 100 or 10, 32)
    print(f"Label array shape: {y_array.shape}")  # Expected: (#samples,)

    # Save arrays
    np.save(path+'/x_'+fn+str(f)+'.npy', data_array)
    np.save(path+'/y_'+fn+str(f)+'.npy', y_array)


def create_folds_from_sampled_data(df,path,insize):
    fiveFold(df, path)

    #create numpy arrays
    for i in range(10):
        print("starting folder: "+str(i))
        train_fold=pd.read_csv(path+"/train"+str(i)+".csv", low_memory=False)
        val_fold=pd.read_csv(path+"/val"+str(i)+".csv", low_memory=False)
        create2Ddata(df,train_fold, i, "train_", path,insize)
        create2Ddata(df,val_fold, i, "val_", path,insize)


insize=10000
path="./kernel3/training_data_100/"
df = pd.read_csv("sampled_data_no_plaques_k3_100of10kx32.csv", low_memory=False)
create_folds_from_sampled_data(df,path,insize)


