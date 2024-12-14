# OCR using classical machine learning and image processing.

### For training following procedure is followed.
1. Download the train and test sets from ICDAR 2003 dataset.
>
    wget http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTrain/char.zip -O train_set.zip
    wget http://www.iapr-tc11.org/dataset/ICDAR2003_RobustReading/TrialTest/char.zip -O test_set.zip

2. Unzip these datasets.
3. Run the analyse_and_clean_dataset.py file to get the dataset in clean and usable format.
3. Run the generate_dataset.py file to get the preprocessed training dataset.
4. Run the train_extra_tree.py file to train an ExtraTreesClassifier on the generated dataset and save the class instance of the same for the inference.

### For the inference following procedure is followed.
1. Follow all the training steps and train the classifier.
2. Run the main.py with --img_path <img_path> and --model_path <model_path> argument to get the prediction about the given image.


### Limitations:
1. Current implementation fails to properly segment individual characters out of word crops.
2. The classifier which is trained on ICDAR 2003 is having 37 % accuracy on test set. This classifier was decided based on LazyPredict run. LazyPredict is a python package which runs all the classifiers on given dataset and give the accuracy values for each of the classifiers.