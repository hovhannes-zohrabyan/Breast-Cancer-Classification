from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from datetime import datetime


from controllers.dataset_controller import DatasetController
from controllers.file_controller import LocalDataController


class VotingModel:

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
        clf_1 = LogisticRegression(multi_class='multinomial', random_state=1)
        clf_2 = RandomForestClassifier(n_estimators=50, random_state=1)
        clf_3 = GaussianNB()
        self.clf = VotingClassifier(estimators=[
            ('LRegression', clf_1), ('Rforest', clf_2), ('GBayes', clf_3)],
            voting='soft', weights=[1, 2, 1],
            flatten_transform=True)
        self.local_data_controller.save_data_pickle('models', 'RandomForest', self.clf)

    def print_accuracy(self):
        X_test, y_test = self.dataset.get_test_data()
        print(self.clf.score(X_test, y_test))
        metrics.plot_roc_curve(self.clf, X_test, y_test)
        plt.show()

    def predict(self, data):
        return self.clf.predict(data)

    def get_model(self):
        return self.clf