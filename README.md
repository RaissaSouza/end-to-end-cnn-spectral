# end-to-end-cnn-spectral
<div align="center">

</div>

<p align="center">
<img src="workflow.png?raw=true">
</p>


Implementation of: "[End-to-end deep learning for identifying Alzheimer’s disease signatures in spectral fluorescence imaging of normal-appearing brain regions in 5xFAD mice]", which is under review (https://doi.org/<coming soon>).

Our code here implements a CNN-based model to classify Alzheimer's disease in normal-appearing tissue of wild-type and 5xFAD mice.

If you find our framework, code, or paper useful to your research, please cite us!
```
@article{
}

```
```

```

### Abstract 
**Purpose**: Alzheimer’s disease (AD) is the leading cause of dementia, characterized by the accumulation of amyloid-β and tau aggregates in the brain. Current diagnostic approaches primarily detect mature plaques, but growing evidence suggests that pre-fibrillar aggregates are present earlier in normal-appearing tissue and may provide a valuable window for intervention. In this work, we propose a novel end-to-end deep learning (DL) model to identify AD signatures directly from normal-appearing tissue (i.e., background parenchyma) of raw spectral fluorescence data.
**Approach**: To develop and evaluate our DL model, we used one million signals extracted from spectral fluorescence images of brain sections collected from wild-type and 5xFAD mice (a transgenic strain carrying five familial human AD mutations).
**Results**: Our results demonstrate that AD-related signatures can indeed be detected in normal-appearing brain regions. Experiments revealed that model performance depends on input dimensions, training set size, and the preprocessing kernel size. The optimal configuration used 1000×32 input, 9,000 training samples, and a 3×3 kernel, achieving >95% accuracy, sensitivity, specificity, F1-score, and area under the receiver operating characteristic curve. Importantly, no mice were misclassified, and saliency analysis indicated that a broad range of spectral fluorescence wavelengths contributed to the model’s decisions.
**Conclusions**: These findings highlight the potential of DL–based analysis of spectral fluorescence data to enable earlier detection of AD-related changes, supporting a shift toward a more proactive strategy for biomarker discovery.
  

## AD classifier
We use the state-of-the-art simple fully convolutional network (SFCN) (doi: 10.1016/J.MEDIA.2020.101871) as our deep learning architecture. The Adam optimizer with an initial learning rate of 0.01, with exponential decay applied after every epoch, and batch size 64 was used during training. The best model (lowest binary cross-entropy testing loss) was saved for evaluation after early stopping with a patience of 10 epochs. 
The code used is in: 
```bash
├── code/sfcn_2D_no_spatial.py

```
To run this code, you will need to change the params variable to match the size of your image data and to read the correct column for your lables. After you update the file, you can save and run:
```
python sfcn_2D_no_spatial.py -x_train ./path_fold_0/x_train_0.npy -y_train ./path_fold_0/y_train_0.npy -x_val ./path_fold_0/x_val_0.npy -y_val ./path_fold_0.npy -out_size_x 1000 -out_size_y 9000 -in_size 1000 -out_name ./path_to_save_model/best_model_fold0 > ./path_to_save_log/fold0.out

```

## Evaluation
The code used for evaluation is in: 
```bash
├── code/inference.py
```
To run the code:

```
python inference.py -x_test ./path_fold_0/x_val_0.npy -y_test ./path_fold_0/y_val_0.npy -fold ./path_file_with_labels/val0.csv -path ./path_to_model/best_model_fold0_best

```

## Saliency maps
The code used to generate the saliency maps is in: 
```bash
├── code/saliency_maps.py
├── code/compress_saliency_maps.py
```
First, generate saliency maps for each test sample per folder:

```
python saliency_maps.py -path ./path_fold_0

```

Then, compress the saliency maps to get a single one per mouse:

```
python compress_saliency_maps.py -path ./path_fold_0

```

## Environment 
Our code for the Keras model pipeline used: 
* Python 3.10.6
* pandas 1.5.0
* numpy 1.23.3
* scikit-learn 1.1.2
* simpleitk 2.1.1.1
* tensorflow-gpu 2.10.0
* cudnn 8.4.1.50
* cudatoolkit 11.7.0

GPU: NVIDIA GeForce RTX 3090



## Resources
* Questions? Open an issue or send an [email](mailto:raissa_souzadeandrad@ucalgary.ca?subject=end-to-end-spectral).
