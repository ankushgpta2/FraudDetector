import pandas as pd 
import numpy as np
import os
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from statsmodels.stats.outliers_influence import variance_inflation_factor
from plot_data import Plotting


class GeneralProcessing():
    def __init__(self):
        self.processed_data = pd.DataFrame()
        # to use for certain general utility functions
        self.ProcessingUtility = ProcessingUtility()
        # for saving the different splits 
        self.base_path_to_use = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    
    def do_general_processing(self, cleaned_data):
        """
        """
        # update the data attribute 
        self.processed_data = cleaned_data

        # filter out previously identified duplicates 
        self.filter_out_duplicates()

        # drop out columns that are completely empty 
        self.processed_data.drop(columns=[field for field in self.processed_data.columns if self.processed_data[field].isna().sum() == len(self.processed_data)], inplace=True)

        # convert bools to integers 
        self.convert_bools_to_ints(self.processed_data)
   
    def filter_out_duplicates(self):
        """
        """
        # filter out the duplicate records 
        self.processed_data = self.processed_data[self.processed_data['is_first_in_group']]
        self.processed_data = self.processed_data[~self.processed_data['isReversal']]
        # drop columns 
        self.processed_data.drop(columns=['is_first_in_group', 'isReversal'], inplace=True)

    def convert_bools_to_ints(self, df):
        """
        """
        # get the boolean columns 
        boolean_columns = [col for col in df.columns if df[col].isin([True, False]).all()]
        # convert boolean columns to integers 
        df[boolean_columns] = df[boolean_columns].apply(self.ProcessingUtility.convert_boolean_column)
        self.processed_data = df
    
    def split_data(self, df):
        """
        """
        # Get unique customerid
        customerid = df['customerId'].unique()

        # Create a DataFrame that contains one row per customerId and whether they have any fraud transactions
        customer_df = df.groupby('customerId')['isFraud'].max().reset_index()

        # Split customerid into training/validation/test
        train_val_customerid, test_customerid = train_test_split(
            customer_df['customerId'], test_size=0.2, stratify=customer_df['isFraud']
        )

        # Split training/validation customerid into training and validation
        train_customerid, val_customerid = train_test_split(
            train_val_customerid, test_size=0.2, stratify=customer_df[customer_df['customerId'].isin(train_val_customerid)]['isFraud']
        )

        # Filter the original DataFrame based on the splits
        train_df = df[df['customerId'].isin(train_customerid)]
        val_df = df[df['customerId'].isin(val_customerid)]
        test_df = df[df['customerId'].isin(test_customerid)]

        # save the dataframes
        if not os.path.isfile(os.path.join(self.base_path_to_use, 'training_split.parquet')):
            train_df.to_parquet(os.path.join(self.base_path_to_use, 'training_split.parquet'), index=False)
        if not os.path.isfile(os.path.join(self.base_path_to_use, 'validation_split.parquet')):
            val_df.to_parquet(os.path.join(self.base_path_to_use, 'validation_split.parquet'), index=False)
        if not os.path.isfile(os.path.join(self.base_path_to_use, 'test_split.parquet')):
            test_df.to_parquet(os.path.join(self.base_path_to_use, 'test_split.parquet'), index=False)
        
        # separate out the y counterparts 
        train, y_train = train_df.drop('isFraud', axis=1), train_df['isFraud']
        val, y_val = val_df.drop('isFraud', axis=1), val_df['isFraud']
        test, y_test = test_df.drop('isFraud', axis=1), test_df['isFraud']

        return train, val, test, y_train, y_val, y_test

    def convert_to_dmatrix(self, training, validation, test):
        """
        """
        # Convert to DMatrix
        d_training = xgb.DMatrix(training.drop('isFraud', axis=1), label=training['isFraud'])
        d_validation = xgb.DMatrix(validation.drop('isFraud', axis=1), label=validation['isFraud'])
        d_test = xgb.DMatrix(test.drop('isFraud', axis=1), label=test['isFraud'])

        return d_training, d_validation, d_test


class FeatureEngineering():
    def __init__(self):
        # initialize some engineering attributes
        self.features_to_one_hot_encode = ['merchantName', 'acqCountry', 'merchantCountryCode', 'posEntryMode', 'posConditionCode',
                                           'merchantCategoryCode', 'transactionType', 'cardPresent']
        self.engineered_dataset = pd.DataFrame()
        # to use the plotting class/methods 
        self.Plotting = Plotting()
        # threshold to use for VIF 
        self.vif_threshold = 10
        # to use for parquet 
        self.base_path_to_use = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        self.path_to_features_dropped_file = os.path.join(self.base_path_to_use, 'transactions_features_dropped.parquet')
        self.path_to_encodings_file = os.path.join(self.base_path_to_use, 'transactions_with_encodings.parquet')
    
    def engineer_features(self, df):
        """
        """
        # create new features 
        self.create_features(df)

        # check for correlations between features
        if not os.path.isfile(self.path_to_features_dropped_file):
            # call function to drop features based on VIF 
            self.drop_high_vif_features(self.engineered_dataset)
            # save the intermediate dataset
            self.engineered_dataset.to_parquet(self.path_to_features_dropped_file, index=False)
        else:
            self.engineered_dataset = pd.read_parquet(self.path_to_features_dropped_file)

        # one hot encode categorical variables 
        if not os.path.isfile(self.path_to_encodings_file):
            # call function for one-hot-encoding
            self.one_hot_encode_features(self.engineered_dataset)
            # save the intermediate dataset
            self.engineered_dataset.to_parquet(self.path_to_encodings_file, index=False)
        else:
            self.engineered_dataset = pd.read_parquet(self.path_to_encodings_file)
    
    def one_hot_encode_features(self, df):
        """
        """
        # initialize encoder
        encoder = OneHotEncoder(sparse_output=False, drop='first')
        
        # fit and transform the specified columns
        cols_to_encode = [x for x in self.features_to_one_hot_encode if x in self.engineered_dataset.columns]
        encoded_columns = encoder.fit_transform(df[cols_to_encode])

        # convert to DataFrame with meaningful column names
        encoded_df = pd.DataFrame(encoded_columns, columns=encoder.get_feature_names_out(cols_to_encode))

        # concatenate with the original DataFrame (drop original columns if necessary)
        self.engineered_dataset = pd.concat([df.drop(columns=cols_to_encode), encoded_df], axis=1)

    def create_features(self, df):
        """
        """
        # convert fields to datetime format
        df['transactionDateTime'] = pd.to_datetime(df['transactionDateTime'])
        df['accountOpenDate'] = pd.to_datetime(df['accountOpenDate'])

        # extract time features for transaction
        df['hour'] = df['transactionDateTime'].dt.hour
        df['day'] = df['transactionDateTime'].dt.day
        df['month'] = df['transactionDateTime'].dt.month
        df['day_of_week'] = df['transactionDateTime'].dt.dayofweek

        # scale the features by standardizing 
        scaler = StandardScaler()
        df[['transactionAmount', 'creditLimit', 'availableMoney']] = scaler.fit_transform(df[['transactionAmount', 'creditLimit', 'availableMoney']])

        # convert to timestamp for transaction time 
        df['timestamp'] = df['transactionDateTime'].astype(int) / 10**9

        # create feature relationships 
        df['transaction_to_credit_ratio'] = df['transactionAmount'] / df['creditLimit']
        df['credit_minus_available'] = df['creditLimit'] - df['availableMoney']
        df['time_since_account_open'] = (df['transactionDateTime'] - df['accountOpenDate']).dt.days
        df['time_since_last_address_change'] = (df['transactionDateTime'] - df['dateOfLastAddressChange']).dt.days
        df['transaction_amount_percentage'] = df['transactionAmount'] / df['availableMoney']
        df['credit_minus_available'] = df['creditLimit'] - df['availableMoney']
        df['time_since_last_transaction'] = df['transactionDateTime'].diff().dt.total_seconds()
        df['transaction_deviation'] = df['transactionAmount'] / df.groupby('accountNumber')['transactionAmount'].transform('median')

        # update the attribute 
        self.engineered_dataset = df 

    def calculate_correlation_matrix(self, df):
        """
        """
        # get only numeric columns
        numeric_df = df.select_dtypes(include=['number'])
        # calculate correlation matrix 
        correlation_matrix = numeric_df.corr()
        # plot the correlation matrix 
        self.Plotting.plot_correlation_matrix(correlation_matrix)
    
    def drop_high_vif_features(self, df):
        """
        Drop features with VIF values above the given threshold until all features have VIF values below the threshold.
        """
        numeric_df = df.select_dtypes(include=['number'])
        numeric_df = numeric_df.dropna()

        while True:
            vif = self.calculate_vif(numeric_df)
            print("Current VIF values:\n", vif)

            # check if all VIF values are below the threshold
            if (vif["VIF"] <= self.vif_threshold).all():
                break
            
            # find the feature with the highest VIF
            feature_to_drop = vif.sort_values(by="VIF", ascending=False).iloc[0]["Feature"]
            print(f"Dropping feature: {feature_to_drop} with VIF: {vif[vif['Feature'] == feature_to_drop]['VIF'].values[0]}")

            # drop the feature with the highest VIF
            numeric_df = numeric_df.drop(columns=[feature_to_drop])
        
        self.engineered_dataset = self.engineered_dataset[numeric_df.columns.to_list()]
    
    def calculate_vif(self, df):
        """
        Calculate the Variance Inflation Factor (VIF) for each feature in the DataFrame.
        """
        vif = pd.DataFrame()
        vif["Feature"] = df.columns
        vif["VIF"] = [variance_inflation_factor(df.values, i) for i in range(df.shape[1])]
        return vif


class ProcessingUtility():
    def __init__(self):
        """
        """

    @staticmethod
    def convert_string_to_numeric(col):
        """
        """
        try:
            return pd.to_numeric(col)
        except ValueError:
            return col
    
    @staticmethod
    def convert_boolean_column(col):
        return col.map({True: 1, False: 0})
