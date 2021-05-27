# Breast Cancer Prediction
from controllers.models.RandomForestModel import RandomForestModel

if __name__ == '__main__':
    model = RandomForestModel()
    model.train_model()
