import requests 
import os 
import shutil
import zipfile 
import logging 
import sys


class LoadData():
    def __init__(self):
        # initialize some constant values 
        self.SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
        self.DATA_DIR = os.path.join(self.SCRIPT_DIR, 'data')
        self.FILE_PATH = os.path.join(self.DATA_DIR, 'transactions.txt')
        self.need_to_unzip = False

    def handle_loading(self):
        # runs the main steps for loading data
        self.check_data_dir()
        self.get_data()

    def check_data_dir(self):
        # check if data directory exists, if not, then create it 
        if not os.path.isdir(self.DATA_DIR): 
            os.makedirs(self.DATA_DIR)
            os.chmod(self.DATA_DIR, 0o755)
            # set the data_dir_created flag to True
            self.data_dir_created = True 

    def get_data(self):
        # check if transactions.txt is not at proper location
        if not os.path.isfile(self.FILE_PATH):
            # check if transactions.txt is one level above + move it
            if os.path.isfile(os.path.join(self.SCRIPT_DIR, 'transactions.txt')):
                shutil.move(os.path.join(self.SCRIPT_DIR, 'transactions.txt'), self.DATA_DIR)
            else:
                # check if transactions.zip is at proper location
                if os.path.isfile(os.path.join(self.DATA_DIR, 'transactions.zip')):
                    self.need_to_unzip = True
                # check if transactions.zip is one level above + move it
                elif os.path.isfile(os.path.join(self.SCRIPT_DIR, 'transactions.zip')):
                    shutil.move(os.path.join(self.SCRIPT_DIR, 'transactions.zip'), self.DATA_DIR)
                    self.need_to_unzip = True
                else:     
                    # need to download the data
                    self.download_repo_zip()
                    self.need_to_unzip = True

            # check if you need to unzip the data
            if self.need_to_unzip:
                with zipfile.ZipFile(os.path.join(self.DATA_DIR, 'transactions.zip'), 'r') as zip_ref:
                    zip_ref.extractall(self.DATA_DIR)
                # provide update
                if os.path.isfile(self.FILE_PATH):
                    print(f"Extracted {os.path.join(self.DATA_DIR, 'transactions.zip')} Successfully!")
                # delete residual zip
                os.remove(os.path.join(self.DATA_DIR, 'transactions.zip'))

    def download_repo_zip(self, owner="CapitalOneRecruiting", repo="DS", branch="master", filename="transactions.zip"):
        """
        Download .zip file from GitHub repository, in order to get data
        """
        response = requests.get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{filename}", stream=True)
        if response.status_code == 200:
            with open(os.path.join(self.DATA_DIR, filename), 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            print(f"\n\nFile downloaded and saved to {os.path.join(self.DATA_DIR, filename)}\n\n")
        else:
            print(f"\n\nFailed to download file. Status code: {response.status_code}\n. To proceed without download, please manually place inside data folder")
