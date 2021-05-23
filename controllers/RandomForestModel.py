from sklearn.ensemble import RandomForestClassifier

from controllers.dataset_controller import DatasetController


class RandomForestModel:

    def __init__(self):
        self.dataset = DatasetController()

    def train_model(self):
        self.clf = RandomForestClassifier(max_depth=2, random_state=0)
        X_train, y_train = self.dataset.get_train_data()
        self.clf.fit(X_train, y_train)

    def print_accuracy(self):
        X_test, y_test = self.dataset.get_test_data()
        print(self.clf.score(X_test, y_test))

    def get_model(self):
        return self.clf