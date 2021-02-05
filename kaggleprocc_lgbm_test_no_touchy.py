import pickle
import pandas as pd
import numpy as np
import gc
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit
import lightgbm as lgb
from sklearn.metrics import (roc_curve, auc, accuracy_score)


# In order to optimize memory usage it is useful to make certain features get certain types
# the dictionary below has been taken from https://www.kaggle.com/theoviel/load-the-totality-of-the-data
dtypes = {
        'MachineIdentifier':                                    'category',
        'ProductName':                                          'category',
        'EngineVersion':                                        'category',
        'AppVersion':                                           'category',
        'AvSigVersion':                                         'category',
        'IsBeta':                                               'int8',
        'RtpStateBitfield':                                     'float16',
        'IsSxsPassiveMode':                                     'int8',
        'DefaultBrowsersIdentifier':                            'float32',
        'AVProductStatesIdentifier':                            'float32',
        'AVProductsInstalled':                                  'float16',
        'AVProductsEnabled':                                    'float16',
        'HasTpm':                                               'int8',
        'CountryIdentifier':                                    'int16',
        'CityIdentifier':                                       'float32',
        'OrganizationIdentifier':                               'float16',
        'GeoNameIdentifier':                                    'float16',
        'LocaleEnglishNameIdentifier':                          'int16',
        'Platform':                                             'category',
        'Processor':                                            'category',
        'OsVer':                                                'category',
        'OsBuild':                                              'int16',
        'OsSuite':                                              'int16',
        'OsPlatformSubRelease':                                 'category',
        'OsBuildLab':                                           'category',
        'SkuEdition':                                           'category',
        'IsProtected':                                          'float16',
        'AutoSampleOptIn':                                      'int8',
        'PuaMode':                                              'category',
        'SMode':                                                'float16',
        'IeVerIdentifier':                                      'float16',
        'SmartScreen':                                          'category',
        'Firewall':                                             'float16',
        'UacLuaenable':                                         'float64', # was 'float32'
        'Census_MDC2FormFactor':                                'category',
        'Census_DeviceFamily':                                  'category',
        'Census_OEMNameIdentifier':                             'float32', # was 'float16'
        'Census_OEMModelIdentifier':                            'float32',
        'Census_ProcessorCoreCount':                            'float16',
        'Census_ProcessorManufacturerIdentifier':               'float16',
        'Census_ProcessorModelIdentifier':                      'float32', # was 'float16'
        'Census_ProcessorClass':                                'category',
        'Census_PrimaryDiskTotalCapacity':                      'float64', # was 'float32'
        'Census_PrimaryDiskTypeName':                           'category',
        'Census_SystemVolumeTotalCapacity':                     'float64', # was 'float32'
        'Census_HasOpticalDiskDrive':                           'int8',
        'Census_TotalPhysicalRAM':                              'float32',
        'Census_ChassisTypeName':                               'category',
        'Census_InternalPrimaryDiagonalDisplaySizeInInches':    'float32', # was 'float16'
        'Census_InternalPrimaryDisplayResolutionHorizontal':    'float32', # was 'float16'
        'Census_InternalPrimaryDisplayResolutionVertical':      'float32', # was 'float16'
        'Census_PowerPlatformRoleName':                         'category',
        'Census_InternalBatteryType':                           'category',
        'Census_InternalBatteryNumberOfCharges':                'float64', # was 'float32'
        'Census_OSVersion':                                     'category',
        'Census_OSArchitecture':                                'category',
        'Census_OSBranch':                                      'category',
        'Census_OSBuildNumber':                                 'int16',
        'Census_OSBuildRevision':                               'int32',
        'Census_OSEdition':                                     'category',
        'Census_OSSkuName':                                     'category',
        'Census_OSInstallTypeName':                             'category',
        'Census_OSInstallLanguageIdentifier':                   'float16',
        'Census_OSUILocaleIdentifier':                          'int16',
        'Census_OSWUAutoUpdateOptionsName':                     'category',
        'Census_IsPortableOperatingSystem':                     'int8',
        'Census_GenuineStateName':                              'category',
        'Census_ActivationChannel':                             'category',
        'Census_IsFlightingInternal':                           'float16',
        'Census_IsFlightsDisabled':                             'float16',
        'Census_FlightRing':                                    'category',
        'Census_ThresholdOptIn':                                'float16',
        'Census_FirmwareManufacturerIdentifier':                'float16',
        'Census_FirmwareVersionIdentifier':                     'float32',
        'Census_IsSecureBootEnabled':                           'int8',
        'Census_IsWIMBootEnabled':                              'float16',
        'Census_IsVirtualDevice':                               'float16',
        'Census_IsTouchEnabled':                                'int8',
        'Census_IsPenCapable':                                  'int8',
        'Census_IsAlwaysOnAlwaysConnectedCapable':              'float16',
        'Wdft_IsGamer':                                         'float16',
        'Wdft_RegionIdentifier':                                'float16',
        'HasDetections':                                        'int8'
        }


print("Loading data")
df_train = pd.read_csv('train.csv', dtype=dtypes)
df_test = pd.read_csv('test.csv', dtype=dtypes)

print("Dropping first columns")
df_train.drop(['Census_OSArchitecture', 'MachineIdentifier'], axis = 1, inplace = True)
df_test.drop(['Census_OSArchitecture', 'MachineIdentifier'], axis = 1, inplace = True)



print("Feature Engineering started")
df_train['HasDetections'] = df_train['HasDetections'].astype('int8')

if 5244810 in df_train.index:
    df_train.loc[5244810,'AvSigVersion'] = '1.273.1144.0'
    df_train['AvSigVersion'].cat.remove_categories('1.2&#x17;3.1144.0',inplace=True)


# ADDING THE ENGINEERED FEATURES
datedictAS = np.load('AvSigVersionTimestamps.npy', allow_pickle=True)[()]
datedictOS = np.load('OSVersionTimestamps.npy', allow_pickle=True)[()]

print("Date features")

df_train['DateAS'] = df_train['AvSigVersion'].map(datedictAS)
df_test['DateAS'] = df_test['AvSigVersion'].map(datedictAS)
df_train['DateOS'] = df_train['Census_OSVersion'].map(datedictOS)
df_test['DateOS'] = df_test['Census_OSVersion'].map(datedictOS)

print("AppVersion2")
df_train['AppVersion2'] = df_train['AppVersion'].map(lambda x: np.int(x.split('.')[1]))
df_test['AppVersion2'] = df_test['AppVersion'].map(lambda x: np.int(x.split('.')[1]))

print("Lagzzz")
df_train['Lag1'] = df_train['DateAS'] - df_train['DateOS']
df_train['Lag1'] = df_train['Lag1'].map(lambda x: x.days//7)
df_test['Lag1'] = df_test['DateAS'] - df_test['DateOS']
df_test['Lag1'] = df_test['Lag1'].map(lambda x: x.days//7)


print("Driveway")
df_train['driveA'] = df_train['Census_SystemVolumeTotalCapacity'].astype('float')/df_train['Census_PrimaryDiskTotalCapacity'].astype('float')
df_test['driveA'] = df_test['Census_SystemVolumeTotalCapacity'].astype('float')/df_test['Census_PrimaryDiskTotalCapacity'].astype('float')
df_train['driveA'] = df_train['driveA'].astype('float32') 
df_test['driveA'] = df_test['driveA'].astype('float32')



df_train['driveB'] = df_train['Census_PrimaryDiskTotalCapacity'].astype('float') - df_train['Census_SystemVolumeTotalCapacity'].astype('float')
df_test['driveB'] = df_test['Census_PrimaryDiskTotalCapacity'].astype('float') - df_test['Census_SystemVolumeTotalCapacity'].astype('float')
df_train['driveB'] = df_train['driveB'].astype('float32') 
df_test['driveB'] = df_test['driveB'].astype('float32')


del df_train['DateAS'], df_train['DateOS']
del df_test['DateAS'], df_test['DateOS']
del datedictAS, datedictOS

gc.collect()

print("Feature Engineering finished")



print("define memory reduction")
def reduce_memory(df, col):
    mx = df[col].max()
    if mx<256:
            df[col] = df[col].astype('uint8')
    elif mx<65536:
        df[col] = df[col].astype('uint16')
    else:
        df[col] = df[col].astype('uint32')
        



ct = 1
   
# removing high cardinality features
for col in df_train.columns.to_list():
    rate = df_train[col].value_counts(normalize=True, dropna=False).values[0]
    if rate > 0.98:
        del df_train[col]
        del df_test[col]
        ct += 1
    
# simple removal
rmv3=['Census_OSSkuName', 'OsVer', 'Census_OSInstallLanguageIdentifier']
rmv4=['SMode']
for col in rmv3+rmv4:
    del df_train[col]
    del df_test[col]
    ct +=1
    
print('Removed',ct,'variables')



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

print("define new factorize")
def factorize(train, test, col):
    print(f"Currently encoding {col}")
    if hasattr(train[col], 'cat'):
        train[col] = train[col].astype('object')
        test[col] = test[col].astype('object')
    encoded_train, uniques = train[col].factorize(sort=True)
    # MAKE SMALLEST LABEL 1, RESERVE 0
    max_encoded_val = encoded_train.max()
    encoded_train = np.where(encoded_train == -1, max_encoded_val + 1, encoded_train)
    train[col] = encoded_train
    encoding_dict = {}
    for encoded_val, previous_val in enumerate(uniques):
        encoding_dict[previous_val] = encoded_val
    print(encoding_dict)
    # possibly non-exhaustvie mapping: 
    # https://stackoverflow.com/questions/42529454/using-map-for-columns-in-a-pandas-dataframe
    test[col].fillna(-1, inplace = True)
    test[col] = test[col].apply(lambda x: max_encoded_val + 2 if x not in uniques and x != -1 else x)
    test[col] = test[col].map(encoding_dict).fillna(test[col])
    # now handling the values which were not in the train set
    # just make them any integer not used already, e.g. max + 2, LGBM doesn't care
    test[col] = np.where(test[col] == -1, max_encoded_val + 1, test[col])
    print("Supposedly finished encoding both sets")
    test[col] = test[col].astype('uint32')


cat_cols = [col for col in df_train.columns if col not in nums and col != "HasDetections"]


print("Encoding features in progress...")
for col in cat_cols:
    factorize(df_train, df_test, col)


print("Data ready for Random Search!")

targets = df_train['HasDetections']
df_train.drop(['HasDetections'], axis = 1, inplace = True)


print("luck'n'loaded")


print("define model")
# Initiate classifier to use
lgb_model = lgb.LGBMClassifier(
          device = "cpu",
          objective = 'binary', 
          n_jobs = -1, 
          silent = True,
          max_depth = -1)


gridParams = {
    'boosting_type': ['gbdt', 'goss'],
    'n_estimators': [20, 40, 100, 150, 200],
    'min_child_samples': list(range(100, 2000, 200)),
    'num_leaves': list(range(10, 150, 20)),
    'subsample': [0.3, 0.25, 0.4, 0.5, 0.6, 0.75, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.3, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    'learning_rate': np.logspace(start=-5, stop=-1, num=8, base=10),
    'reg_alpha': [0, 1e-2, 1e-1, 1, 5, 10, 100]
    }


print("definiujemy RGS")

N_ITERATIONS = 100

radom = RandomizedSearchCV(
                    estimator = lgb_model,
                    param_distributions = gridParams, 
                    n_iter = N_ITERATIONS,
                    verbose = 0, 
                    cv = ShuffleSplit(n_splits=10, 
                                      test_size=0.01, 
                                      train_size=0.1),
                    n_jobs = -1,
                    scoring = 'roc_auc',
                    refit = True,
                        )

print("run RGS")

# Run the grid
radom.fit(df_train,
         targets, 
         **{'categorical_feature' : cat_cols})

# Print the best parameters found
print(f'Best score reached: {radom.best_score_}')
print(f'Best parameters: {radom.best_params_}')


print("save model")


with open('test_no_touchy_kaggle_preprocess_RSCV_lgbm.pickle', 'wb') as handle:
    pickle.dump(radom, handle, protocol=pickle.HIGHEST_PROTOCOL)


######## FITTING A WHOLE MODEL ON THESE PARAMETERS ###########

print("Fit modelu na najlepszych parametrach z RGS")

kaggle_params = radom.best_params_

kaggle_lgbm = lgb.LGBMClassifier(
          device = "cpu",
          objective = 'binary', 
          n_jobs = -1, 
          silent = True,
          max_depth = -1)
kaggle_lgbm.set_params(**kaggle_params)

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

with open('test_no_touchy_kaggle_preprocess_lgbm.pickle', 'wb') as handle:
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
sub_df.to_csv('test_no_touchy_kaggle_preprocess_lgbm_submission.csv', index=False)

print("Job's done")
