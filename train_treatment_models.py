import multiprocessing as mp
from copy import copy

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, GridSearchCV
from sklearn.model_selection._search import ParameterGrid

from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier

from xbcausalforest import XBCF
# Monkeypatch XBCF class to to datapreprocessing automatically
class myXBCF(XBCF):
    def fit(self, x_t, x, y, z, p_cat=0):
        z= z.astype('int32')

        self.sdy = np.std(y)
        y = y - np.mean(y)
        y = y / self.sdy

        super().fit(x_t, x, y, z, p_cat=p_cat)

        return self


    def predict(self, *args,**kwargs):
        tauhats = super().predict(*args,**kwargs)
        
        b = self.b.transpose()

        thats = self.sdy * tauhats * (b[1] - b[0])
        thats_mean = np.mean(thats[:, self.get_params()['burnin']:], axis=1)
        
        return thats_mean

import sys
sys.path.append('/Users/hauptjoh/projects/research/treatment-learn')

from treatlearn.double_robust_transformation import DoubleRobustTransformer
from treatlearn.indirect import SingleModel, HurdleModel, TwoModelRegressor
from treatlearn.evaluation import transformed_outcome_loss

def predict_treatment_models(X, y, c, g, tau_conversion, tau_basket, tau_response, split, fold_index):

    treatment_model_lib = {}
    conversion_model_lib = {}
    #regression_model_lib = {}
    N_JOBS=1


    params = {
    "gbtr":
        {"learning_rate" : 0.01,
        "n_estimators" : 100,
        "max_depth" : 3,
        "subsample" : 0.95,
        'min_samples_leaf':0.01,
    },
    'gbtr_2ndstage' : {
        'learning_rate':[0.05, 0.075, 0.1, 0.125, 0.15], 
        'max_depth':[2,3,4],
        'n_estimators':[100],
        'subsample':[0.95],
        #'max_features':[0.9],
        'min_samples_leaf':[1, 50, 100],
        },
    "gbtc":
        {"learning_rate" : 0.01,
        "n_estimators" : 100,
        "max_depth" : 3,
        "subsample" : 0.95,
        'min_samples_leaf':0.01,
    },
        "rf":{
        "n_estimators" : 500,
        "min_samples_leaf" : 50
    },
    "reg":{
         "alpha":10
    },
    "logit":{
        "C":1,
        'solver':'liblinear',
        'max_iter':1000
    },
    'xbcf':{
        'num_sweeps':100,
        'burnin':20,
        'num_trees_pr':100,
        'num_trees_trt':50,
        'num_cutpoints':100,
        'alpha_pr':0.95,
        'beta_pr':2,
        'alpha_trt':0.95,
        'beta_trt':2,        
    }
    }

    # Tuning grids when tuning is enabled
    param_grid = {
    'gbtr' : {
        'learning_rate':[0.05, 0.075, 0.1, 0.125, 0.15], 
        'max_depth':[2,3,4],
        'n_estimators':[100],
        'subsample':[0.95],
        #'max_features':[0.9],
        'min_samples_leaf':[1, 50, 100],
        },
    'gbtc' : {
        'learning_rate':[0.05, 0.075, 0.1, 0.125, 0.15], 
        'max_depth':[2,3,4],
        'n_estimators':[100],
        'subsample':[0.95],
        #'max_features':[0.9],
        'min_samples_leaf':[1, 50, 100],
        },
    'rf' : {
        'n_estimators':[500],
        'min_samples_leaf': [50],
        'max_features': [0.05,0.1,0.15],
    },
    "reg":{
         "alpha":[10, 2, 1, 0.5, 0.25, 0.166, 0.125, 0.1]
    },
    "logit":{
        "C":[0.1,0.5,1,2,4,6,8,10],
        'solver':['liblinear'],
        'max_iter':[1000]
    }}


    # Find columns that are not binary with max=1
    num_columns = np.where(X.columns[(X.max(axis=0) != 1)])[0].tolist()
    n_cat = (X.max(axis=0) == 1).sum()

    # Split the train and validation data
    X_val, y_val, c_val, g_val, tau_conversion_val, tau_basket_val, tau_response_val = [obj.to_numpy().astype(float)[split[1]] for obj in [X, y, c, g, tau_conversion, tau_basket, tau_response]]
    X, y, c, g, tau_conversion, tau_basket, tau_response = [obj.to_numpy().astype(float)[split[0]] for obj in [X, y, c, g, tau_conversion, tau_basket, tau_response]]


    # Normalize the data
    ct = ColumnTransformer([
        # (name, transformer, columns)
        # Transformer for categorical variables
        #("onehot",
        #     OneHotEncoder(categories='auto', handle_unknown='ignore', ),
        #     cat_columns),
        # Transformer for numeric variables
        ("num_preproc", StandardScaler(), num_columns)
        ],
        # what to do with variables not specified in the indices?
        remainder="passthrough")

    X = ct.fit_transform(X)
    X_val = ct.transform(X_val)

    # Treatment indicator as variable
    Xg = np.c_[X, g]
    Xg_val = np.c_[X_val, g_val]

    # Double robust transformation
    # TODO: Cross fitting to avoid overfitting the nuisance models during model tuning
    # Note that the final model evaluaton on X_val is unaffected
    dr = DoubleRobustTransformer()
    y_dr = dr.fit_transform(X, y, g)
    y_dr.mean()

    #### Parameter Tuning  ####
    # Cross-validation folds stratified randomization by (outcome x treatment group)
    splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
    cg_groups = 2*g+c # Groups 0-4 depending on combinations [0,1]x[0,1]
    folds = list(splitter.split(X, cg_groups))
    folds_c1 = list(splitter.split(X[c==1,:], g[c==1]))

    ## Simple GBT predictors
    # 
    cv = GridSearchCV(GradientBoostingRegressor(), param_grid["gbtr"], scoring='neg_mean_squared_error', n_jobs=N_JOBS, verbose=0, cv=folds)
    cv.fit(X, y)
    params["gbtr"] = cv.best_params_
    print(f"gbtr params: {cv.best_params_}")

    cv = GridSearchCV(GradientBoostingClassifier(), param_grid["gbtc"], scoring='neg_brier_score', n_jobs=N_JOBS, verbose=0, cv=folds)
    cv.fit(X, c)
    params["gbtc"] = cv.best_params_
    print(f"gbtc params: {cv.best_params_}")

    cv = GridSearchCV(GradientBoostingRegressor(), param_grid["gbtr"], scoring='neg_mean_squared_error', n_jobs=N_JOBS, verbose=0, cv=folds_c1)
    cv.fit(X[c==1,:], y[c==1])
    params["gbtr_2ndstage"] = cv.best_params_
    print(f"gbtr_2ndstage params: {cv.best_params_}")

    ## Simple linear predictors
    cv = GridSearchCV(Ridge(), param_grid["reg"], scoring='neg_mean_squared_error', n_jobs=N_JOBS, verbose=0, cv=folds)
    cv.fit(X, y)
    params["reg"] = cv.best_params_
    print(f"reg params: {cv.best_params_}")

    cv = GridSearchCV(LogisticRegression(solver='liblinear', max_iter=1000), param_grid["logit"], scoring='neg_brier_score', n_jobs=N_JOBS, verbose=0, cv=folds)
    cv.fit(X, c)
    params["logit"] = cv.best_params_
    print(f"logit params: {cv.best_params_}")

    cv = GridSearchCV(Ridge(), param_grid["reg"], scoring='neg_mean_squared_error', n_jobs=N_JOBS, verbose=0, cv=folds_c1)
    cv.fit(X, y)
    params["reg_2ndstage"] = cv.best_params_
    print(f"reg_2ndstage params: {cv.best_params_}")
    

    def grid_search_cv(X, y, g, estimator, param_grid, folds, **kwargs):
        list_param_grid = list(ParameterGrid(param_grid))
        list_param_loss = []
        for param in list_param_grid:
            list_split_loss = []
            for split in folds:
                    # Split the train and validation data
                    _estimator = copy(estimator)
                    X_val, y_val, g_val = [obj[split[1]] for obj in [X, y, g]]
                    X_train, y_train, g_train = [obj[split[0]] for obj in [X, y, g]]
                    _estimator.set_params(**param)
                    _estimator.fit(X=X_train,y=y_train,g=g_train, **{name:value[split[0]] for name,value in kwargs.items()})
                    pred = _estimator.predict(X_val)
                    tol = transformed_outcome_loss(pred, y_val, g_val)
                    list_split_loss.append(tol)
            list_param_loss.append(np.mean(list_split_loss))
        return list_param_grid[list_param_loss.index(min(list_param_loss))]


    # Recommended defaults from XBCF paper
    params['xbcf']['tau_pr'] = 0.1/params['xbcf']['num_trees_pr'] # 0.1 * var(y_norm) = 0.1
    params['xbcf']['tau_trt'] = 0.1/params['xbcf']['num_trees_trt']
    params['xbcf']['mtry_pr'] = int(X.shape[1])
    params['xbcf']['mtry_trt'] = int(X.shape[1])
    print(f"xbcf params: {params['xbcf']}")

    #### Single Model ####
    ## GBT regression
    single_gbt_regressor = SingleModel(GradientBoostingRegressor(**params["gbtr"]))
    # Tune single model based on Transformed Outcome Loss
    best_params = grid_search_cv(X, y, g, single_gbt_regressor, param_grid['gbtr'], folds)
    single_gbt_regressor.set_params(**best_params)
    print(f"single model GBT params: {best_params}")

    single_gbt_regressor.fit(X, y, g=g)
    treatment_model_lib['single-model_outcome_gbt'] = single_gbt_regressor


    ## RF regression
    # single_rf_regressor = SingleModel(RandomForestRegressor(**params["rf"]))
    # single_rf_regressor.fit(X, y, g=g)

    # treatment_model_lib['single_model_rf'] = single_rf_regressor


    ## Hurdle Random Forest
    # conversion_rf = RandomForestClassifier(**params["rf"])
    # conversion_rf.fit(Xg, c)

    # basket_rf = RandomForestRegressor(**params["rf"])                                  
    # basket_rf.fit(Xg[c==1], y[c==1], sample_weight=1/conversion_rf.predict_proba(Xg)[c==1,1])

    # treatment_model_lib['hurdle_rf'] = SingleModel(HurdleModel(conversion_classifier=conversion_rf, value_regressor=basket_rf))


    ## Hurdle Gradient Boosting
    single_hurdle_gbt = SingleModel(HurdleModel(conversion_classifier=GradientBoostingClassifier(**params["gbtc"]), 
                            value_regressor=GradientBoostingRegressor(**params["gbtr_2ndstage"])))
    #best_params = grid_search_cv(X=X, y=y, g=g, c=c, estimator=single_hurdle_gbt, param_grid=param_grid['gbtr'], folds=folds)
    #print(f"single model Hurdle GBT params: {best_params}")
    # -> Tuning while fixing same parameters for both models of the hurdle does not give good results

    single_hurdle_gbt.fit(X=X,y=y,c=c,g=g)
    treatment_model_lib['single-model_hurdle_gbt'] = single_hurdle_gbt


    #### Two-Model Approach ####
    ## Linear regression
    two_model_reg0 = Ridge(**params['reg'])
    two_model_reg0.fit(X[g==0,:], y[g==0])
    two_model_reg1 = Ridge(**params['reg'])
    two_model_reg1.fit(X[g==1,:], y[g==1])
    treatment_model_lib["two-model_outcome_linear"] = TwoModelRegressor(control_group_model=two_model_reg0, treatment_group_model=two_model_reg1)

    # ## Gradient Boosting Regression
    two_model_gbt0 = GradientBoostingRegressor(**params["gbtr"])
    two_model_gbt0.fit(X[g==0,:], y[g==0])
    two_model_gbt1 = GradientBoostingRegressor(**params["gbtr"])
    two_model_gbt1.fit(X[g==1,:], y[g==1])

    treatment_model_lib["two-model_outcome_gbt"] = TwoModelRegressor(control_group_model=two_model_gbt0, treatment_group_model=two_model_gbt1)

    # ## Random Forest Regression
    # two_model_rf0 = RandomForestRegressor(**params["rf"])    
    # two_model_rf0.fit(X[g==0,:], y[g==0])

    # two_model_rf1 = RandomForestRegressor(**params["rf"])    
    # two_model_rf1.fit(X[g==1,:], y[g==1])

    # treatment_model_lib["two_model_rf"] = TwoModelRegressor(control_group_model=two_model_rf0, treatment_group_model=two_model_rf1)

    
    ## Hurdle Linear model
    conversion_logit0 = LogisticRegression(**params['logit'])
    conversion_logit0.fit(X[g==0,:], c[g==0])

    basket_reg0 = Ridge(**params['reg_2ndstage'])                                 
    basket_reg0.fit(X[(c==1) & (g==0),:], y[(c==1) & (g==0)],sample_weight=1/conversion_logit0.predict_proba(X)[(c==1) & (g==0), 1].clip(0.05,0.95)) # 

    hurdle_reg0 = HurdleModel(conversion_classifier=conversion_logit0, value_regressor=basket_reg0)

    conversion_logit1 = LogisticRegression(**params['logit'])
    conversion_logit1.fit(X[g==1,:], c[g==1])

    basket_reg1 = Ridge(**params['reg_2ndstage'])                                 
    basket_reg1.fit(X[(c==1) & (g==1),:], y[(c==1) & (g==1)], sample_weight=1/conversion_logit1.predict_proba(X)[(c==1) & (g==1), 1].clip(0.05,0.95)) # 
    
    hurdle_reg1 = HurdleModel(conversion_classifier=conversion_logit1, value_regressor=basket_reg1)

    treatment_model_lib["two-model_hurdle_linear"] = TwoModelRegressor(control_group_model=hurdle_reg0, treatment_group_model=hurdle_reg1)


    ## Hurdle Random Forest
    # conversion_rf0 = RandomForestClassifier(**params["rf"])
    # conversion_rf0.fit(X[g==0,:], c[g==0])

    # basket_rf0 = RandomForestRegressor(**params["rf"])                                  
    # basket_rf0.fit(X[(c==1) & (g==0),:], y[(c==1) & (g==0)], 
    #             sample_weight=1/conversion_rf0.predict_proba(X)[(c==1) & (g==0), 1])

    # hurdle_rf0 = HurdleModel(conversion_classifier=conversion_rf0, value_regressor=basket_rf0)

    # conversion_rf1 = RandomForestClassifier(**params["rf"])
    # conversion_rf1.fit(X[g==1,:], c[g==1])

    # basket_rf1 = RandomForestRegressor(**params["rf"])                                  
    # basket_rf1.fit(X[(c==1) & (g==1),:], y[(c==1) & (g==1)], 
    #             sample_weight=1/conversion_rf1.predict_proba(X)[(c==1) & (g==1), 1])
    # hurdle_rf1 = HurdleModel(conversion_classifier=conversion_rf1, value_regressor=basket_rf1)

    # treatment_model_lib["two_model_hurdle_rf"] = TwoModelRegressor(control_group_model=hurdle_rf0, treatment_group_model=hurdle_rf1)

    ## Hurdle GBT
    two_model_hurdle_gbt = TwoModelRegressor(control_group_model=HurdleModel(
                                                conversion_classifier=GradientBoostingClassifier(**params["gbtc"]),
                                                value_regressor=GradientBoostingRegressor(**params["gbtr_2ndstage"]) ), 
                                             treatment_group_model=HurdleModel(
                                                 conversion_classifier=GradientBoostingClassifier(**params["gbtc"]),
                                                value_regressor=GradientBoostingRegressor(**params["gbtr_2ndstage"]) 
                                             ))

    two_model_hurdle_gbt.fit(X=X, y=y, g=g, c=c)
    treatment_model_lib["two-model_hurdle_gbt"] = two_model_hurdle_gbt


    #### Double robust ####

    # Regression
    cv = GridSearchCV(Ridge(), param_grid['reg'], scoring='neg_mean_squared_error', n_jobs=N_JOBS, verbose=0, cv=folds)
    cv.fit(X, y_dr)
    print(f"DR Ridge params: {cv.best_params_}")
    treatment_model_lib["dr_outcome_linear"] = Ridge(**cv.best_params_)
    treatment_model_lib["dr_outcome_linear"].fit(X, y_dr)

    # GBT
    cv = GridSearchCV(GradientBoostingRegressor(), param_grid['gbtr'], scoring='neg_mean_squared_error', n_jobs=N_JOBS, verbose=0, cv=folds)
    cv.fit(X, y_dr)
    print(f"DR GBT params: {cv.best_params_}")
    treatment_model_lib["dr_outcome_gbt"] = GradientBoostingRegressor(**cv.best_params_)
    treatment_model_lib["dr_outcome_gbt"].fit(X, y_dr)

    #### XBCF ####
    treatment_model_lib["xbcf_outcome_xbcf"] = myXBCF(**params["xbcf"])
    treatment_model_lib["xbcf_outcome_xbcf"].fit(x_t=X, x=X, y=y, z=g, p_cat=int(n_cat))


    ##### Conversion Models ####
    conversion_model_lib["single-model_outcome_linear"] = LogisticRegression(**params['logit'])
    conversion_model_lib["single-model_outcome_linear"].fit(X[g == 1, :], c[g == 1])

    conversion_model_lib["single-model_outcome_gbt"] = GradientBoostingClassifier(**params["gbtc"])
    conversion_model_lib["single-model_outcome_gbt"].fit(X[g == 1, :], c[g == 1])

    # conversion_model_lib["rf"] = RandomForestClassifier(**params["rf"])
    # conversion_model_lib["rf"].fit(X[g==1,:], c[g==1])


    # ### Evaluation
    # ##### Conversion treatment effect

    treatment_conversion_train = {}
    treatment_conversion_test  = {}

    treatment_conversion_train["single-model_hurdle_gbt"] = treatment_model_lib['single-model_hurdle_gbt'].model.conversion_classifier.predict_proba( np.c_[X,     np.ones((X.shape[0],1))] )[:,1]     - treatment_model_lib['single-model_hurdle_gbt'].model.conversion_classifier.predict_proba(np.c_[X, np.zeros((X.shape[0],1))])[:,1]
    treatment_conversion_test["single-model_hurdle_gbt"]  = treatment_model_lib['single-model_hurdle_gbt'].model.conversion_classifier.predict_proba( np.c_[X_val, np.ones((X_val.shape[0],1))] )[:,1] - treatment_model_lib['single-model_hurdle_gbt'].model.conversion_classifier.predict_proba(np.c_[X_val, np.zeros((X_val.shape[0],1))])[:,1]

    treatment_conversion_train["two-model_hurdle_linear"] = treatment_model_lib['two-model_hurdle_linear'].treatment_group_model.predict_hurdle(X) - treatment_model_lib['two-model_hurdle_linear'].control_group_model.predict_hurdle(X)
    treatment_conversion_test["two-model_hurdle_linear"] = treatment_model_lib['two-model_hurdle_linear'].treatment_group_model.predict_hurdle(X_val) - treatment_model_lib['two-model_hurdle_linear'].control_group_model.predict_hurdle(X_val)

    treatment_conversion_train["two-model_hurdle_gbt"] = treatment_model_lib['two-model_hurdle_gbt'].treatment_group_model.predict_hurdle(X) - treatment_model_lib['two-model_hurdle_gbt'].control_group_model.predict_hurdle(X)
    treatment_conversion_test["two-model_hurdle_gbt"] = treatment_model_lib['two-model_hurdle_gbt'].treatment_group_model.predict_hurdle(X_val) - treatment_model_lib['two-model_hurdle_gbt'].control_group_model.predict_hurdle(X_val)
    
    # treatment_conversion_train["hurdle_rf"] = treatment_model_lib['hurdle_rf'].model.conversion_classifier.predict_proba( np.c_[X,     np.ones((X.shape[0],1))] )[:,1]     - treatment_model_lib['hurdle_rf'].model.conversion_classifier.predict_proba(np.c_[X, np.zeros((X.shape[0],1))])[:,1]
    # treatment_conversion_test["hurdle_rf"]  = treatment_model_lib['hurdle_rf'].model.conversion_classifier.predict_proba( np.c_[X_val, np.ones((X_val.shape[0],1))] )[:,1] - treatment_model_lib['hurdle_rf'].model.conversion_classifier.predict_proba(np.c_[X_val, np.zeros((X_val.shape[0],1))])[:,1]

    treatment_conversion_train["ATE__"] = (c[g==1].mean())-(c[g==0].mean()) * np.ones([X.shape[0]])
    treatment_conversion_test["ATE__"] =   (c[g==1].mean())-(c[g==0].mean()) * np.ones([X_val.shape[0]])

    # Baselines
    treatment_conversion_train["oracle__"] = tau_conversion
    treatment_conversion_test["oracle__"] =  tau_conversion_val


    # ##### Basket value treatment effect

    treatment_basketvalue_train = {}
    treatment_basketvalue_test  = {}

    treatment_basketvalue_train["single-model_hurdle_gbt"] = treatment_model_lib['single-model_hurdle_gbt'].model.value_regressor.predict( np.c_[X,     np.ones((X.shape[0],1))] )     - treatment_model_lib['single-model_hurdle_gbt'].model.value_regressor.predict(np.c_[X, np.zeros((X.shape[0],1))])
    treatment_basketvalue_test["single-model_hurdle_gbt"]  = treatment_model_lib['single-model_hurdle_gbt'].model.value_regressor.predict( np.c_[X_val, np.ones((X_val.shape[0],1))] ) - treatment_model_lib['single-model_hurdle_gbt'].model.value_regressor.predict(np.c_[X_val, np.zeros((X_val.shape[0],1))])

    treatment_basketvalue_train["two-model_hurdle_linear"] = treatment_model_lib['two-model_hurdle_linear'].treatment_group_model.predict_value(X) - treatment_model_lib['two-model_hurdle_linear'].control_group_model.predict_value(X)
    treatment_basketvalue_test["two-model_hurdle_linear"] = treatment_model_lib['two-model_hurdle_linear'].treatment_group_model.predict_value(X_val) - treatment_model_lib['two-model_hurdle_linear'].control_group_model.predict_value(X_val)

    treatment_basketvalue_train["two-model_hurdle_gbt"] = treatment_model_lib['two-model_hurdle_gbt'].treatment_group_model.predict_value(X) - treatment_model_lib['two-model_hurdle_gbt'].control_group_model.predict_value(X)
    treatment_basketvalue_test["two-model_hurdle_gbt"] = treatment_model_lib['two-model_hurdle_gbt'].treatment_group_model.predict_value(X_val) - treatment_model_lib['two-model_hurdle_gbt'].control_group_model.predict_value(X_val)
    
    treatment_basketvalue_train["ATE__"] = (y[(c==1) & (g==1)].mean())-(y[(c==1) & (g==0)].mean()) * np.ones([X.shape[0]])
    treatment_basketvalue_test["ATE__"] =   (y[(c==1) & (g==1)].mean())-(y[(c==1) & (g==0)].mean()) * np.ones([X_val.shape[0]])

    treatment_basketvalue_train["oracle__"] =  tau_basket
    treatment_basketvalue_test["oracle__"] =   tau_basket_val

    # ##### Treatment response prediction

    treatment_pred_train = {key: model.predict(X) for key, model in treatment_model_lib.items()}
    treatment_pred_test = {key: model.predict(X_val) for key, model in treatment_model_lib.items()}

    # Baselines
    treatment_pred_train["oracle__"] = tau_response
    treatment_pred_test["oracle__"] =   tau_response_val

    treatment_pred_train["ATE__"] = (y[g==1].mean())-(y[g==0].mean()) * np.ones([X.shape[0]])
    treatment_pred_test["ATE__"] =   (y[g==1].mean())-(y[g==0].mean()) * np.ones([X_val.shape[0]])


    ############ Conversion C(T=1) prediction
    conversion_pred_train = {key: model.predict_proba(X)[:,1]   for key, model in conversion_model_lib.items()}
    conversion_pred_val = {key: model.predict_proba(X_val)[:,1] for key, model in conversion_model_lib.items()}

    conversion_pred_train["single-model_hurdle_gbt"] =   treatment_model_lib['single-model_hurdle_gbt'].model.conversion_classifier.predict_proba( np.c_[X, np.ones((X.shape[0],1))] )[:,1]
    conversion_pred_val["single-model_hurdle_gbt"] =     treatment_model_lib['single-model_hurdle_gbt'].model.conversion_classifier.predict_proba( np.c_[X_val, np.ones((X_val.shape[0],1))])[:,1]

    #conversion_pred_train["hurdle_rf"] =   treatment_model_lib['hurdle_rf'].model.conversion_classifier.predict_proba( np.c_[X, np.ones((X.shape[0],1))] )[:,1]
    #conversion_pred_val["hurdle_rf"] =     treatment_model_lib['hurdle_rf'].model.conversion_classifier.predict_proba( np.c_[X_val, np.ones((X_val.shape[0],1))])[:,1]

    conversion_pred_train["two-model_hurdle_linear"] = treatment_model_lib["two-model_hurdle_linear"].treatment_group_model.conversion_classifier.predict_proba(X)[:,1]
    conversion_pred_val["two-model_hurdle_linear"] = treatment_model_lib["two-model_hurdle_linear"].treatment_group_model.conversion_classifier.predict_proba(X_val)[:,1]

    #conversion_pred_train["two_model_hurdle_rf"] = treatment_model_lib["two_model_hurdle_rf"].treatment_group_model.conversion_classifier.predict_proba(X)[:,1]
    #conversion_pred_val["two_model_hurdle_rf"] = treatment_model_lib["two_model_hurdle_rf"].treatment_group_model.conversion_classifier.predict_proba(X_val)[:,1]

    conversion_pred_train["two-model_hurdle_gbt"] = treatment_model_lib["two-model_hurdle_gbt"].treatment_group_model.conversion_classifier.predict_proba(X)[:,1]
    conversion_pred_val["two-model_hurdle_gbt"] = treatment_model_lib["two-model_hurdle_gbt"].treatment_group_model.conversion_classifier.predict_proba(X_val)[:,1]

    conversion_pred_train["Conversion-Rate__"] = np.ones(X.shape[0]) * c[g==1].mean()
    conversion_pred_val["Conversion-Rate__"] =   np.ones(X_val.shape[0]) * c[g==1].mean()

    ## Output formatting
    return({"train":{"idx": split[0],
        "conversion": conversion_pred_train,
        "treatment_conversion": treatment_conversion_train,
        "treatment_basket_value": treatment_basketvalue_train,
        "treatment_spending": treatment_pred_train,
        "params":params
    }, 
            "test":{"idx": split[1],
        "conversion": conversion_pred_val,
        "treatment_conversion": treatment_conversion_test,
        "treatment_basket_value": treatment_basketvalue_test,
        "treatment_spending": treatment_pred_test
    }
            })



#### Script
if __name__ == "__main__":
    
    DEBUG = False

    # Load the data

    X = pd.read_csv("../data/fashionB_clean_nonlinear.csv")
    #X = data.copy()

    # PARAMETERS
    SEED=42
    N_SPLITS = 5
    np.random.seed(SEED)

    # Downsampling for debugging
    if DEBUG is True:
        X = X.sample(5000)

    c = X.pop('converted')
    g = X.pop('TREATMENT')
    y = X.pop('checkoutAmount')
    tau_conversion = X.pop('TREATMENT_EFFECT_CONVERSION')
    tau_basket = X.pop('TREATMENT_EFFECT_BASKET')
    tau_response = X.pop('TREATMENT_EFFECT_RESPONSE')


    # Cross-validation folds stratified randomization by (outcome x treatment group)
    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    cg_groups = 2*g+c # Groups 0-4 depending on combinations [0,1]x[0,1]
    folds = list(splitter.split(X, cg_groups))
    
    library_predictions = []
    def log_result(x):
        try:
            library_predictions.append(x)
        except Exception as e:
            library_predictions.append(str(e))
    
    if DEBUG is True:
        temp = predict_treatment_models(X, y, c, g, tau_conversion, tau_basket, tau_response, folds[0], 1)
        print(temp)
    else:
        pool = mp.Pool(N_SPLITS)

        for i,fold in enumerate(folds):
            pool.apply_async(predict_treatment_models, 
                args=(X, y, c, g, tau_conversion, tau_basket, tau_response, fold, i), 
                callback = log_result)
        pool.close()
        pool.join()
        print("Cross-Validation complete.")

        #with open("../results/treatment_model_predictions.json", "w") as outfile:
        #    json.dump(library_predictions, outfile)
        np.save( "../results/treatment_model_predictions", library_predictions, allow_pickle=True)

    print("Done!")

