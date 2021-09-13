from itertools import product
import multiprocessing as mltp
from copy import deepcopy

import numpy as np
import pandas as pd

import scoring

class GridSearchTotal:
    def __init__(self,estimator=None,investigate={},model_params={},exercise="C"):
        self._estimator = estimator
        self._investigate = investigate
        self._model_params = model_params
        self._exercise = exercise # C --> classification, R --> Regression
        
        self._combinations =  [{**inv, **model_params} for inv in self._create_combinations(investigate)]
        self.report_ = pd.DataFrame(None,index=range(len(self._combinations)),columns=investigate.keys())
    
    def __repr__(self):
        return f"GridSearch({self._estimator}, {self._investigate}, {self._model_params})"
    
    def _create_combinations(self,param_dict):
        combinations = []
        for i in product(*param_dict.values()):
            expl = dict()
            for j,k in enumerate(i):
                expl[list(param_dict.keys())[j]] = k
            combinations.append(expl)
        return combinations
    
    def _explore_one(self,pack):
        mdl = pack["model"](**pack["parameter"]).fit(pack["X_tr"],pack["y_tr"]) # training
        y_pred = mdl.predict(pack["X_ts"]) # predict
        if self._exercise == "C":
            scores = scoring.evaluation(pack["y_ts"],y_pred)
        elif self._exercise == "R":
            scores = scoring.evaluation(scoring.reg_to_binary(pack["y_ts"]), scoring.reg_to_binary(y_pred))
        return {"C": getattr(mdl,"C"), "gamma": getattr(mdl,"gamma"), **dict(scores)}
    
    
    def fit_predict(self,X_train,X_test,y_train,y_test):
        # create the packs
        packs = []
        for i in range(len(self._combinations)):
            packs.append({"model":deepcopy(self._estimator),
                          "parameter": self._combinations[i],
                          "X_tr": X_train, "y_tr": y_train,
                          "X_ts": X_test, "y_ts": y_test})
        
        # launch investigation
        pool = mltp.Pool(processes=4)
        res = pool.map(self._explore_one,packs)
        pool.close()
        
        self.report_ = pd.DataFrame(res)
        return self
    
    def get_report(self,by=None,ascending=True):
        if by is None:
            return self.report_
        else:
            return self.report_.sort_values(by=by,ascending=ascending)