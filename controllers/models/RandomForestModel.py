from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn import metrics


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
            print("Model is not trained, starting training")
            self.train_model()

    def train_model(self):
        print("Model training started at ", datetime.now())
        self.clf = RandomForestClassifier(max_depth=2, random_state=0)
        X_train, y_train = self.dataset.get_train_data()
        self.clf.fit(X_train, y_train)
        self.local_data_controller.save_data_pickle('models', 'RandomForest', self.clf)

    def print_accuracy(self):
        X_test, y_test = self.dataset.get_test_data()
        print('Accuracy for Random Forest Model', self.clf.score(X_test, y_test), sep='\n')
        print('Confusion Matrix for Random Forest Model', metrics.confusion_matrix(y_test, self.clf.predict(X_test)), sep='\n')
        metrics.plot_roc_curve(self.clf, X_test, y_test)
        plt.show()

    def predict(self, data):
        return self.clf.predict(data)

    def get_model(self):
        return self.clf