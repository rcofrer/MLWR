import pandas as pd
import numpy as np
import pickle


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



df_train['HasDetections'] = df_train['HasDetections'].astype('int8')



print("define memory reduction")
def reduce_memory(df, col):
    mx = df[col].max()
    if mx<256:
            df[col] = df[col].astype('uint8')
    elif mx<65536:
        df[col] = df[col].astype('uint16')
    else:
        df[col] = df[col].astype('uint32')
        

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
    # possibly non-exhaustvie mapping: 
    # https://stackoverflow.com/questions/42529454/using-map-for-columns-in-a-pandas-dataframe
    test[col].fillna(-1, inplace = True)
    test[col] = test[col].apply(lambda x: max_encoded_val + 2 if x not in uniques and x != -1 else x)
    test[col] = test[col].map(encoding_dict).fillna(test[col])
    # now handling the values which were not in the train set
    # just make them any integer not used already, e.g. max + 2, LGBM doesn't care
    test[col] = np.where(test[col] == -1, max_encoded_val + 1, test[col])
    test[col] = test[col].astype('uint32')


cat_cols = [col for col in df_train.columns if col not in nums and col != "HasDetections"]


print("Encoding features and reducing memory usage in progress...")
for col in cat_cols:
    factorize(df_train, df_test, col)
    reduce_memory(df_train, col)
    reduce_memory(df_test, col)


with open('proper_integer_encoded_datas.pickle', 'wb') as handle:
    pickle.dump((df_train, df_test), handle, protocol=pickle.HIGHEST_PROTOCOL)

print("Job's done")
