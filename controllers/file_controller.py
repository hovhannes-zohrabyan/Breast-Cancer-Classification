import pickle
import os


class LocalDataController:

    @staticmethod
    def save_data_pickle(directory, name, data):
        abs_directory = os.path.abspath("Saved_Data/" + directory + "/" + name + ".pickle")
        doc = open(abs_directory, "wb")
        pickle.dump(data, doc)
        doc.close()

    @staticmethod
    def read_data_pickle(directory, name):
        abs_directory = os.path.abspath("../Saved_Data/" + directory + "/" + name + ".pickle")
        print(abs_directory)
        if os.path.isfile(abs_directory):
            pickle_file = open(abs_directory, "rb")
            classifier = pickle.load(pickle_file)
            pickle_file.close()
        else:
            raise FileNotFoundError

        return classifier
