import requests
import zipfile  # import the zipfile module to extract files from the downloaded zip file
import os  # import the os module for creating and managing directories and file paths # import the requests module for downloading the zip file from a URL
import pandas as pd  # import pandas for working with CSV files
import glob  # import glob for finding all CSV files in a directory

# import train_test_split from sklearn for splitting the data into training and testing sets
import datetime

# URL for the .zip file to be downloaded
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00501/PRSA2017_Data_20130301-20170228.zip"

# Paths for the directories where the .zip file and its contents will be stored
path = "./data/"
# Name of the .zip file
file_name = "Airqualitydata.zip"

# Full path to the .zip file
file_path = os.path.join(path, file_name)

# Here we check if the directory "path" exists, and if it doesn't, we create it using os.mkdir()
try:
    os.stat(path)
except OSError:
    os.mkdir(path)

# We use the requests module to download the file, and then write the content to a file in the specified directory using open()
print("Downloading data ---------------------")
r = requests.get(url, allow_redirects=True)
open(file_path, "wb").write(r.content)
print("File downloaded...")

# We use the zipfile module to extract the contents of the zip file to the specified directory using extractall()
with zipfile.ZipFile(file_path, "r") as zip_ref:
    zip_ref.extractall(path)
print("File extracted to", path)

# Read all CSV files in the directory, merge them into a single DataFrame, and print the shape of the resulting DataFrame
df = pd.concat(
    [
        pd.read_csv(f)
        for f in glob.glob(os.path.join(path + "\PRSA_Data_20130301-20170228", "*"))
    ],
    ignore_index=True,
)
print(df.shape)

# Adding a date variable set it as an index and sort the data by index in order to keep the time structure
# maybe this part could be added to the preprocessing.py
df["date"] = df.apply(
    lambda x: datetime.datetime(x["year"], x["month"], x["day"], x["hour"]), axis=1
)
df = df.sort_values(["date"], ascending=True)
df.set_index("date", inplace=True)


# Save the merged data to a new CSV file in the specified directory, and then delete the original zip file
df.to_csv(os.path.join(path, "data_merged.csv"), index=False)
os.remove(file_path)
print("Data extracted...")
