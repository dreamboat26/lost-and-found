
import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.preprocessing import StandardScaler

np.set_printoptions(suppress=True)
np.random.seed(42)


class ExtraTreeClassifier:
    def __init__(self, label_set, img_h=30, img_w=17, use_pca=True):
        self.use_pca = use_pca
        self.img_h = img_h
        self.img_w = img_w
        if self.use_pca:
            self.sc = StandardScaler()
            self.pca = PCA(n_components=0.90)

        self.label_set = label_set
        self.label_to_cls = {}
        self.cls_to_label = {}
        for i, c in enumerate(self.label_set):
            self.label_to_cls[i] = c
            self.cls_to_label[c] = i

    def __preprocess(self, data):
        if data.shape[2] == 3:
            data = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)

        data = cv2.resize(data, (self.img_w, self.img_h))
        data = data.reshape(1, -1)
        data = np.float32(data)
        return data

    def fit(self, train_x, train_y):
        assert train_x.shape[0] == train_y.shape[0]
        assert train_y.shape[1] == 1

        if self.use_pca:
            train_x = self.sc.fit_transform(train_x)
            train_x = self.pca.fit_transform(train_x)
            explained_variance = self.pca.explained_variance_ratio_
            print("Variance explained: ", explained_variance)
            print("Number of dimensions: ", len(explained_variance))

        self.feature_size = train_x.shape[1]
        self.extra_tree = ExtraTreesClassifier(n_estimators=1000, random_state=42)
        self.extra_tree.fit(train_x, train_y)

    def predict(self, test_img, preprocess=True):
        if preprocess:
            test_img = self.__preprocess(test_img)

        if self.use_pca:
            test_img = self.sc.transform(test_img)
            test_img = self.pca.transform(test_img)

        detected_class_id = self.extra_tree.predict(test_img)[0]
        return self.label_to_cls[detected_class_id]
