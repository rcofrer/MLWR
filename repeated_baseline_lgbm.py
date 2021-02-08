import pickle
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import lightgbm as lgb
from sklearn.metrics import (roc_curve, auc, accuracy_score)

print("Loading data")

with open('proper_integer_encoded_datas.pickle', 'rb') as handle:
    dataset = pickle.load(handle)

df_train = dataset[0]
df_test = dataset[1]


targets = df_train['HasDetections']
df_train.drop(['HasDetections'], axis = 1, inplace = True)


nums = [
    'Census_ProcessorCoreCount',
    'Census_PrimaryDiskTotalCapacity',
    'Census_SystemVolumeTotalCapacity',
    'Census_TotalPhysicalRAM',
    'Census_InternalPrimaryDiagonalDisplaySizeInInches',
    'Census_InternalPrimaryDisplayResolutionHorizontal',
    'Census_InternalPrimaryDisplayResolutionVertical',
    'Census_InternalBatteryNumberOfCharges',
    'Census_OSBuildNumber',
    'Census_OSBuildRevision',
    'DriveA',
    'DriveB',
    'Lag1'
    ]

cat_cols = [col for col in df_train.columns if col not in nums and col != "HasDetections"]

print("Fit the baseline model using default LGBM parameters")

kaggle_lgbm = lgb.LGBMClassifier(
          device = "cpu",
          objective = 'binary', 
          n_jobs = -1, 
          silent = True,
          max_depth = -1)

x_train, x_val, y_train, y_val = train_test_split(df_train, targets, 
                                                test_size=0.15, stratify=targets,
                                                  random_state = 42)

kaggle_lgbm.fit(x_train, y_train,
                early_stopping_rounds = 100,
                eval_set=[(x_val, y_val)],
                feature_name = x_train.columns.to_list(),
                categorical_feature = cat_cols,
                eval_metric = 'auc', 
                verbose = 0)

with open('baseline_lgbm.pickle', 'wb') as handle:
    pickle.dump(kaggle_lgbm, handle, protocol=pickle.HIGHEST_PROTOCOL)

del x_train, x_val, y_train, y_val
del targets
del df_train
gc.collect()

######## SUBMISSION ##########

print("Submission")

probs = kaggle_lgbm.predict_proba(df_test)

sub_df = pd.read_csv("sample_submission.csv")
sub_df['HasDetections'] = probs[:, 1]
sub_df.to_csv('baseline_lgbm_submission.csv', index=False)

print("Job's done")
