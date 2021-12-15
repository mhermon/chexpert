# University of Iowa CS:5430 Fall 2021 Project, CheXpert

This is the code for our CS:5430 Fall 2021 Project where we tackled the CheXpert competition

Requirements (what we know works):  
python 3.8  
scipy 1.4.1   
pytorch 1.10.0  
scikit-learn 0.24.2   
tensorflow 2.3.0  
libauc 1.1.6  
numpy 1.18.5   

To get predictions:

```
python3 python test_features.py --input <PATH_TO_NPY_TEST_HIDDEN_FEATURES> --model_paths ./models/class_0_#2 ./models/class_1_#2 ./models/class_2_#2 ./models/class_3_#2 ./models/class_4_#2 --output ./predictions.npy --ensemble
```

To train a model:

```
python train_features.py -tf <PATH_TO_TRAIN_FEATURES> -tl <PATH_TO_TRAIN_LABELS> -vf <PATH_TO_VALIDATION_FEATURES> -vl <PATH_TO_VALIDATION_LABELS> --num_epochs=5 --label_smoothing=smart --model_name=<MODEL_NAME> --learning_rate=0.1
