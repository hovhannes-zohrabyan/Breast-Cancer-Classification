# Breast Cancer Prediction
from controllers.models.RandomForestModel import RandomForestModel
from controllers.voting_model.VotingClassifierModel import VotingModel

if __name__ == '__main__':
    model = RandomForestModel()
    # model.train_model()
    model.print_accuracy()

    vote_model = VotingModel()
    vote_model.print_accuracy()
