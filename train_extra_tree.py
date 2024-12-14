import pickle
import string

import numpy as np

from training.classifier import ExtraTreeClassifier

np.set_printoptions(suppress=True)
np.random.seed(42)

label_set = string.digits + string.ascii_uppercase + string.ascii_lowercase

avg_aspect_ratio = 1.75
IMG_H = 30
IMG_W = int(IMG_H / avg_aspect_ratio)

dataset_path = "./training/dataset_v2.npy"
dataset = np.load(dataset_path)
dataset = np.float32(dataset)
dataset = dataset[:, 0, :]
print("Dataset shape:", dataset.shape)

np.random.shuffle(dataset)

split_size = int(0.8 * len(dataset))
train_x = dataset[:split_size, :-1]
train_y = dataset[:split_size, -1:]

test_x = dataset[split_size:, :-1]
test_y = dataset[split_size:, -1:]

clf_model = ExtraTreeClassifier(label_set, IMG_H, IMG_W, True)
clf_model.fit(train_x, train_y)

counter = 0
for i in range(len(test_x)):
    data = test_x[i : i + 1, :]
    pred = clf_model.predict(data, preprocess=False)
    gt = clf_model.label_to_cls[test_y[i][0]]
    if pred == gt:
        counter += 1

acc = counter / len(test_x)
print(f"Acc: {acc:.4f}")


model_output_path = "./training/extra_trees_classifier.pkl"
pickle.dump(clf_model, open(model_output_path, "wb"))
