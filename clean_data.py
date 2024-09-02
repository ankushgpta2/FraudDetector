import os
import re
import pandas as pd 
import numpy as np


class CleanData():
    def __init__(self):
        self.beginning_delim_pattern = r'^[^A-Za-z]+'
        self.ending_delim_pattern = r'[^A-Za-z]+$'
        self.unwanted_chars = ["{", "}", "'", '"']
        self.final_lines = pd.DataFrame()
        self.unique_number_of_fields = []
        self.base_path_to_use = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')

    def clean_raw_data(self):
        # first check if a previous parquet has been saved, if so, then load this in and skip everything
        path_to_parquet = os.path.join(self.base_path_to_use, 'transactions.parquet')
        if os.path.isfile(path_to_parquet):
            self.final_lines = pd.read_parquet(path_to_parquet)
            self.unique_number_of_fields = [len(self.final_lines.columns.to_list())]
            return

        # read file + ensure that the downloaded data is not empty 
        with open(os.path.join(self.base_path_to_use, 'transactions.txt')) as file:
            lines = file.readlines()

        if lines:
            temp_dict_list = []
            # loop through each line + perform certain operations
            for line in lines:
                # clean up each line
                line = line.strip()
                # remove unwanted characters 
                line = self.cleanup_line(line)
                # store information into dictionary --> list
                if line:
                    split_line = re.split(r',|:\s', line)
                    temp_dict = {}
                    for x in range(0, len(split_line), 2):
                        # convert the value
                        converted_val = self.convert_value(split_line[x+1].strip())
                        temp_dict[split_line[x].strip()] = converted_val
                    temp_dict_list.append(temp_dict)
                    # get number of fields and update list 
                    if len(temp_dict.keys()) not in self.unique_number_of_fields:
                        self.unique_number_of_fields.append(len(temp_dict.keys()))
                
            # convert to pandas dataframe + concat
            self.final_lines = pd.DataFrame(temp_dict_list)

            # Replace empty strings with NaN
            self.final_lines.replace("", np.nan, inplace=True)

            # save final dataframe as parquet 
            self.final_lines.to_parquet(path_to_parquet, index=False)

    
    def cleanup_line(self, line):
        for x in self.unwanted_chars:
            line = line.replace(x, '')
        return line

    @staticmethod
    def convert_value(value):
        # Convert to boolean if the value matches 'true'/'false' (case-insensitive)
        if isinstance(value, str):
            value_lower = value.lower()
            if value_lower in ['true', 'false']:
                return value_lower == 'true'
            # Try to convert to numeric
            try:
                return pd.to_numeric(value, errors='raise')
            except ValueError:
                return value
        return value