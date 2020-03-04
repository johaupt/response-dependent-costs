
import multiprocessing as mp

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, StratifiedKFold, GridSearchCV

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier

#import xbart

import sys
sys.path.append('/Users/hauptjoh/projects/treatment-learn')

from treatlearn.double_robust_transformation import DoubleRobustTransformer
from treatlearn.indirect import SingleModel, HurdleModel, TwoModelRegressor

def predict_treatment_models(X, y, c, g, tau_conversion, tau_basket, tau_response, split, fold_index):

    treatment_model_lib = {}
    conversion_model_lib = {}
    regression_model_lib = {}
    N_JOBS=1


    params = {"gbt":
        {"learning_rate" : 0.05,
        "n_estimators" : 100,
        "max_depth" : 2,
        "subsample" : 0.7,
        "n_iter_no_change" : 10,
        "validation_fraction" : 0.1
    },
        "rf":{
        "n_estimators" : 500,
        "min_samples_leaf" : 50
    },
         "reg":{
         "C":1
    },
    "logit":{
        "C":1
    }}

    # Tuning grids when tuning is enabled
    param_grid = {'gbt' : {
    'learning_rate':[0.01,0.025,0.05],
    'max_depth':[2,3,4],
    'n_estimators':[100,200,400,600,800],
    'n_iter_no_change':[100],
    'subsample':[0.8]
        },
    'rf' : {
        'n_estimators':[500],
        'min_samples_leaf': [50],
        'max_features': [0.05,0.1,0.15],
    },
    "reg":{
         "C":[0.1,0.5,1,2,4,6,8,10]
    },
    "logit":{
        "C":[0.1,0.5,1,2,4,6,8,10]
    }}


    # Find columns that are not binary with max=1
    num_columns = np.where(X.columns[(X.max(axis=0) != 1)])[0].tolist()

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
    # TODO: Cross fitting to avoid overfitting the nuisance models
    dr = DoubleRobustTransformer()
    y_dr = dr.fit_transform(X, y, g)
    y_dr.mean()


    #### Parameter Tuning  ####
    cv = GridSearchCV(GradientBoostingRegressor(), param_grid["gbt"], 'neg_mean_squared_error', n_jobs=N_JOBS, verbose=0, cv=5)
    cv.fit(X, y)
    params[model] = cv.best_params_
        
    # cv = GridSearchCV(treatment_model_lib[model], param_grid[model], 'neg_mean_squared_error', n_jobs=N_JOBS, verbose=0, cv=5)
    # cv.fit(X, y_dr)
    # treatment_model_lib[model].set_params(**cv.best_params_)

    cv = GridSearchCV(LinearRegression(), param_grid["reg"], 'neg_mean_squared_error', n_jobs=N_JOBS, verbose=0, cv=5)
    cv.fit(X, y)
    params["reg"] = cv.best_params_

    cv = GridSearchCV(LogisticRegression(), param_grid["logit"], 'neg_mean_squared_error', n_jobs=N_JOBS, verbose=0, cv=5)
    cv.fit(X, c)
    params["logit"] = cv.best_params_

    #### Single Model ####
    ## GBT regression
    single_gbt_regressor = SingleModel(GradientBoostingRegressor(**params["gbt"]))
    single_gbt_regressor.fit(X, y, treatment_group=g)

    treatment_model_lib['single-model_outcome_gbt'] = single_gbt_regressor


    ## RF regression
    # single_rf_regressor = SingleModel(RandomForestRegressor(**params["rf"]))
    # single_rf_regressor.fit(X, y, treatment_group=g)

    # treatment_model_lib['single_model_rf'] = single_rf_regressor


    ## Hurdle Random Forest
    # conversion_rf = RandomForestClassifier(**params["rf"])
    # conversion_rf.fit(Xg, c)

    # basket_rf = RandomForestRegressor(**params["rf"])                                  
    # basket_rf.fit(Xg[c==1], y[c==1], sample_weight=1/conversion_rf.predict_proba(Xg)[c==1,1])

    # treatment_model_lib['hurdle_rf'] = SingleModel(HurdleModel(conversion_classifier=conversion_rf, value_regressor=basket_rf))


    ## Hurdle Gradient Boosting
    # Conversion classification
    conversion_gbt = GradientBoostingClassifier(**params["gbt"])
    conversion_gbt.fit(Xg, c)

    # Basket value (given purchase) regression
    basket_gbt = GradientBoostingRegressor(**params["gbt"])                                  
    basket_gbt.fit(Xg[c==1], y[c==1])#, sample_weight=1/conversion_gbt.predict_proba(Xg)[c==1,1])

    treatment_model_lib['single-model_hurdle_gbt'] = SingleModel(HurdleModel(conversion_classifier=conversion_gbt, value_regressor=basket_gbt))


    #### Two-Model Approach ####
    ## Linear regression
    two_model_reg0 = LinearRegression()
    two_model_reg0.fit(X[g==0,:], y[g==0])
    two_model_reg1 = LinearRegression()
    two_model_reg1.fit(X[g==1,:], y[g==1])
    treatment_model_lib["two-model_outcome_linear"] = TwoModelRegressor(control_group_model=two_model_reg0, treatment_group_model=two_model_reg1)

    # ## Gradient Boosting Regression
    two_model_gbt0 = GradientBoostingRegressor(**params["gbt"])
    two_model_gbt0.fit(X[g==0,:], y[g==0])
    two_model_gbt1 = GradientBoostingRegressor(**params["gbt"])
    two_model_gbt1.fit(X[g==1,:], y[g==1])

    treatment_model_lib["two-model_outcome_gbt"] = TwoModelRegressor(control_group_model=two_model_gbt0, treatment_group_model=two_model_gbt1)

    # ## Random Forest Regression
    # two_model_rf0 = RandomForestRegressor(**params["rf"])    
    # two_model_rf0.fit(X[g==0,:], y[g==0])

    # two_model_rf1 = RandomForestRegressor(**params["rf"])    
    # two_model_rf1.fit(X[g==1,:], y[g==1])

    # treatment_model_lib["two_model_rf"] = TwoModelRegressor(control_group_model=two_model_rf0, treatment_group_model=two_model_rf1)

    
    ## Hurdle Linear model
    conversion_logit0 = LogisticRegression(penalty="l2", C=1, solver='liblinear', max_iter=200)
    conversion_logit0.fit(X[g==0,:], c[g==0])

    basket_reg0 = LinearRegression()                                 
    basket_reg0.fit(X[(c==1) & (g==0),:], y[(c==1) & (g==0)]) # ,sample_weight=1/conversion_logit0.predict_proba(X)[(c==1) & (g==0), 1]

    hurdle_reg0 = HurdleModel(conversion_classifier=conversion_logit0, value_regressor=basket_reg0)

    conversion_logit1 = LogisticRegression(penalty="l2", C=1, solver='liblinear', max_iter=200)
    conversion_logit1.fit(X[g==1,:], c[g==1])

    basket_reg1 = LinearRegression()                                 
    basket_reg1.fit(X[(c==1) & (g==1),:], y[(c==1) & (g==1)]) # , sample_weight=1/conversion_logit1.predict_proba(X)[(c==1) & (g==1), 1]
    
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
    conversion_gbt0 = GradientBoostingClassifier(**params["gbt"])
    conversion_gbt0.fit(X[g == 0,:], c[g == 0])

    basket_gbt0 = GradientBoostingRegressor(**params["gbt"])                                  
    basket_gbt0.fit(X[(c == 1) & (g == 0), :], y[(c == 1) & (g == 0)]) #, sample_weight=1/conversion_gbt0.predict_proba(X)[(c==1) & (g==0), 1]

    hurdle_gbt0 = HurdleModel(conversion_classifier=conversion_gbt0, value_regressor=basket_gbt0)

    conversion_gbt1 = GradientBoostingClassifier(**params["gbt"])
    conversion_gbt1.fit(X[g == 1, :], c[g == 1])

    basket_gbt1 = GradientBoostingRegressor(**params["gbt"])                                
    basket_gbt1.fit(X[(c==1) & (g==1),:], y[(c==1) & (g==1)]) #, sample_weight=1/conversion_gbt1.predict_proba(X)[(c==1) & (g==1), 1]
    hurdle_gbt1 = HurdleModel(conversion_classifier=conversion_gbt1, value_regressor=basket_gbt1)

    treatment_model_lib["two-model_hurdle_gbt"] = TwoModelRegressor(control_group_model=hurdle_gbt0, treatment_group_model=hurdle_gbt1)


    #### Double robust ####

    # Regression
    treatment_model_lib["dr_outcome_linear"] = LinearRegression()
    treatment_model_lib["dr_outcome_linear"].fit(X, y_dr)

    # GBT
    treatment_model_lib["dr_outcome_gbt"] = GradientBoostingRegressor(**params["gbt"])
    treatment_model_lib["dr_outcome_gbt"].fit(X, y_dr)


    ##### Conversion Models ####
    conversion_model_lib["single-model_outcome_linear"] = LogisticRegression(penalty="l2", C=1, solver='liblinear', max_iter=200)
    conversion_model_lib["single-model_outcome_linear"].fit(X[g == 1, :], c[g == 1])

    conversion_model_lib["single-model_outcome_gbt"] = GradientBoostingClassifier(**params["gbt"])
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

if __name__ == "__main__":
    
    # Load the data

    X = pd.read_csv("../data/fashionB_clean_linear.csv")
    #X = data.copy()

    # PARAMETERS
    SEED=42
    N_SPLITS = 5
    np.random.seed(SEED)

    # Downsampling for debugging
    #X = X.sample(5000)

    c = X.pop('converted')
    g = X.pop('TREATMENT')
    y = X.pop('checkoutAmount')
    tau_conversion = X.pop('TREATMENT_EFFECT_CONVERSION')
    tau_basket = X.pop('TREATMENT_EFFECT_BASKET')
    tau_response = X.pop('TREATMENT_EFFECT_RESPONSE')


    # splitter = StratifiedShuffleSplit(n_splits=2, test_size=0.25, random_state=123)
    # idx_train, idx_test = next(splitter.split(X,g))

    splitter = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    cg_groups = 2*g+c # Groups 0-4 depending on combinations [0,1]x[0,1]
    folds = list(splitter.split(X, cg_groups))
    
    library_predictions = []
    def log_result(x):
        try:
            library_predictions.append(x)
        except Exception as e:
            library_predictions.append(str(e))
    
    temp = predict_treatment_models(X, y, c, g, tau_conversion, tau_basket, tau_response, folds[0], 1)
    print(temp)

    # pool = mp.Pool(3)

    # for i,fold in enumerate(folds):
    #     pool.apply_async(predict_treatment_models, 
    #         args=(X, y, c, g, tau_conversion, tau_basket, tau_response, fold, i), 
    #         callback = log_result)
    # pool.close()
    # pool.join()
    # print("Cross-Validation complete.")

    # #with open("../results/treatment_model_predictions.json", "w") as outfile:
    # #    json.dump(library_predictions, outfile)
    # np.save( "../results/treatment_model_predictions", library_predictions, allow_pickle=True)

    print("Done!")

