from sklearn.ensemble import RandomForestClassifier

from controllers.dataset_controller import DatasetController
from controllers.file_controller import LocalDataController


class RandomForestModel:

    def __init__(self):
        self.dataset = DatasetController()
        self.local_data_controller = LocalDataController()
        try:
            self.clf = self.local_data_controller.read_data_pickle('models', 'RandomForest')
            print('Model is already trained, do not train it')
        except FileNotFoundError:
            print("Model is not trained, please train it")

    def train_model(self):
        self.clf = RandomForestClassifier(max_depth=2, random_state=0)
        X_train, y_train = self.dataset.get_train_data()
        self.clf.fit(X_train, y_train)
        self.local_data_controller.save_data_pickle('models', 'RandomForest', self.clf)

    def print_accuracy(self):
        X_test, y_test = self.dataset.get_test_data()
        print(self.clf.score(X_test, y_test))
    #     TODO: Add ROC AUC curves

    def predict(self, data):
        return self.clf.predict(data)

    def get_model(self):
        return self.clf