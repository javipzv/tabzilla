from pyAgrum.skbn import BNClassifier
import numpy as np
from models.basemodel import BaseModel

class TANModel(BaseModel):
    def __init__(self, params, args):
        super().__init__(params, args)
        self.model = BNClassifier(**params)

    @classmethod
    def define_trial_parameters(cls, trial, args):
        params = {
            "priorWeight": trial.suggest_float("priorWeight", 0.5, 1, step=0.1),
            "discretizationNbBins": trial.suggest_int("discretizationNbBins", 2, 12),
        }
        return params
    
    @classmethod
    def get_random_parameters(cls, seed):
        rs = np.random.RandomState(seed)
        params = {
            "priorWeight": rs.uniform(0.1, 1.0),
            "discretizationNbBins": rs.randint(2, 20),
        }
        return params
    
    @classmethod
    def default_parameters(cls):
        params = {
            "priorWeight": 1,
            "discretizationNbBins": 5,
        }
        return params
    
    def get_classes(self):
        return self.model.classes_