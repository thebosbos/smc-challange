

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

import pickle




# load dataset
uncleaned_data = pd.read_csv('C:\Users\bouss\Desktop\smc\smc-challange\model2\data.csv')

# remove timestamp from dataset (always first column)
uncleaned_data = uncleaned_data.iloc[: , 1:]
data = pd.DataFrame()

# keep track of which columns are categorical and what 
# those columns' value mappings are
# structure: {colname1: {...}, colname2: {...} }
cat_value_dicts = {}
final_colname = uncleaned_data.columns[len(uncleaned_data.columns) - 1]

# for each column...
for (colname, colval) in uncleaned_data.iteritems():

  # check if col is already a number; if so, add col directly
  # to new dataframe and skip to next column
  if isinstance(colval.values[0], (np.integer, float)):
    data[colname] = uncleaned_data[colname].copy()
    continue

  # structure: {0: "lilac", 1: "blue", ...}
  new_dict = {}
  val = 0 # first index per column
  transformed_col_vals = [] # new numeric datapoints

  # if not, for each item in that column...
  for (row, item) in enumerate(colval.values):
    
    # if item is not in this col's dict...
    if item not in new_dict:
      new_dict[item] = val
      val += 1
    
    # then add numerical value to transformed dataframe

    transformed_col_vals.append(new_dict[item])
  
  # reverse dictionary only for final col (0, 1) => (vals)
  if colname == final_colname:
    new_dict = {value : key for (key, value) in new_dict.items()}

  cat_value_dicts[colname] = new_dict
  data[colname] = transformed_col_vals


### -------------------------------- ###
###           model training         ###
### -------------------------------- ###

# select features and predicton; automatically selects last column as prediction
cols = len(data.columns)
num_features = cols - 1
x = data.iloc[: , :num_features]
y = data.iloc[: , num_features:]

# split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25)

# instantiate the model (using default parameters)
model = LogisticRegression()
model.fit(x_train, y_train.values.ravel())
y_pred = model.predict(x_test)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(model, open(filename, 'wb'))
