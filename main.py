import load_data, clean_data, explore_data, plot_data, process_data
from models import xgboost
import os 


# initialize all required classes/methods 
LoadData = load_data.LoadData()
CleanData = clean_data.CleanData()
ExploreData = explore_data.DataExploration()
FindDuplicates = explore_data.DuplicateTransactions()
GeneralProcessing = process_data.GeneralProcessing()
FeatureEngineering = process_data.FeatureEngineering()
XGBoost = xgboost.XGBoost()
PlotData = plot_data.Plotting()

# call the load data functions/methods --> retrieve data 
LoadData.handle_loading()

# clean the raw data 
CleanData.clean_raw_data()

# conduct initial exploration on the data 
ExploreData.initial_exploration(
    cleaned_data=CleanData.final_lines,
    unique_number_of_fields=CleanData.unique_number_of_fields
)

# generate histogram of the processed amounts of each transaction
# PlotData.plot_histogram(CleanData.final_lines['transactionAmount'])

# identify reversed/multi-swipe duplicate transactions 
FindDuplicates.find_dup_transactions(CleanData.final_lines)

# process the data before passing into downstream model (XGBoost)
GeneralProcessing.do_general_processing(
    cleaned_data=FindDuplicates.cleaned_data_dup_ids
)

# do some feature engineering 
FeatureEngineering.engineer_features(GeneralProcessing.processed_data)

# split the dataset 
data_splits = GeneralProcessing.split_data(FeatureEngineering.engineered_dataset)

# call XGBoost for tuning/training 
XGBoost.tune_and_train(
    training=data_splits[0],
    evaluation=data_splits[1],
    y_train=data_splits[3],
    y_val=data_splits[4]
)

# call XGBoost for evaluation 
XGBoost.predict(
    test=data_splits[2],
    y_test=data_splits[5]
)
 

