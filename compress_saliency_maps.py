import numpy as np
import pandas as pd
import os
import argparse
import sys


#parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('-path', type=str, help='path')
args = parser.parse_args()

def compress_saliency_maps(path):
    for fold in range(10):

        saliency_folder = f"{path}/saliency_maps_fold_{fold}/"
        saliency_files = sorted([f for f in os.listdir(saliency_folder) if f.endswith('.npy')])

        all_reduced_saliency = []

        for file_name in saliency_files:
            file_path = os.path.join(saliency_folder, file_name)
            saliency_map = np.load(file_path)  # shape: (1, 1000, 32, 1)
            
            # Remove batch and channel dimensions → shape becomes (1000, 32)
            saliency_2d = saliency_map[0].squeeze()
            
            # Average over the 1000 rows → shape becomes (32,)
            reduced_saliency = np.mean(saliency_2d, axis=0)
            
            all_reduced_saliency.append(reduced_saliency)

        # Convert to array → shape (num_samples, 32)
        all_reduced_saliency = np.array(all_reduced_saliency)

        # Average across all samples → shape (32,)
        final_saliency = np.mean(all_reduced_saliency, axis=0)
        print(final_saliency)
        # Save final 1x32 average saliency vector
        np.save(os.path.join(saliency_folder, f'average_saliency_fold_{fold}.npy'), final_saliency)

        print("Final averaged saliency vector saved.")
    return

def combine_averaged_saliency_scores(path):
    saliency_folder = path+'/saliency_maps_fold_'
    all_data = []

    for fold in range(10):  # folds 0 to 10
        file_path = os.path.join(saliency_folder + str(fold), f'average_saliency_fold_{fold}.npy')
        
        # Load the 1x32 saliency vector
        saliency_vector = np.load(file_path)  # shape: (32,)
        
        # Prepare row: fold number + 32 values
        row = [fold] + saliency_vector.tolist()
        all_data.append(row)

    # Create column names: fold, s0, s1, ..., s31
    columns = ['fold'] + [f's{i}' for i in range(32)]

    # Create DataFrame
    df = pd.DataFrame(all_data, columns=columns)
    print(df)

    # Ensure folder exists
    output_folder = path
    os.makedirs(output_folder, exist_ok=True)

    # Save CSV
    csv_path = os.path.join(output_folder, 'average_compressed_saliency_all_folds.csv')
    df.to_csv(csv_path, index=False)

    print(f"Combined average saliency CSV saved to: {csv_path}")
    return

path=args.path
#compress_saliency_maps(path)
combine_averaged_saliency_scores(path)
