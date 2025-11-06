from interpret_extension.glassbox import NAMClassifier
import numpy as np
from models.basemodel import BaseModel

class NAMModel(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)
        self.model = NAMClassifier(n_jobs=-1, **params)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "num_epochs": trial.suggest_int("num_epochs", 10, 40, step=10),
            "num_learners": trial.suggest_int("num_learners", 1, 5, step=1),
        }
        return params

    @classmethod
    def get_random_parameters(cls, seed):
        rs = np.random.RandomState(seed)
        params = {
            "num_epochs": rs.choice([10, 25, 35]),
            "num_learners": rs.choice([1, 3, 5]),
        }
        return params
    
    @classmethod
    def default_parameters(cls):
        params = {
            "num_epochs": 15,
            "num_learners": 3,
        }
        return params
    
    def get_classes(self):
        return self.model.classes_