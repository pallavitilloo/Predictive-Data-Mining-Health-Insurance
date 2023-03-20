import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import auc, roc_curve, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# Divide columns - numeric and categorical
num_feat = ['Age', 'Vintage']
cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year', 'Vehicle_Age_gt_2_Years',
            'Vehicle_Damage_Yes', 'Region_Code', 'Policy_Sales_Channel']

# Test data
train = pd.read_csv('HI_Pred_train.csv', skiprows=range(1, 100000), nrows=10)

# Divide columns - numeric and categorical
num_feat = ['Age', 'Vintage']
cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year', 'Vehicle_Age_gt_2_Years',
            'Vehicle_Damage_Yes', 'Region_Code', 'Policy_Sales_Channel']

# Preprocessing begins

# Gender can be made as categorical - Female = 0 and Male = 1
train['Gender'] = train['Gender'].map({'Female': 0, 'Male': 1}).astype(int)

# Get categorical variables - Vehicle age and Vehicle Damage; drop the first category as it is redundant
train = pd.get_dummies(train, drop_first=True)

# Rename column names containing special characters '</>' and set their type as integer
train = train.rename(
    columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
train['Vehicle_Age_lt_1_Year'] = train['Vehicle_Age_lt_1_Year'].astype('int')
train['Vehicle_Age_gt_2_Years'] = train['Vehicle_Age_gt_2_Years'].astype('int')
train['Vehicle_Damage_Yes'] = train['Vehicle_Damage_Yes'].astype('int')

# Drop ID column as it is not required
train = train.drop('id', axis=1)
train = train.drop('Response', axis=1)

# Store all categorical columns as String
for column in cat_feat:
    train[column] = train[column].astype('str')

# Data Prediction begins
filename = 'rf_bal_model.sav'
rf_load = pickle.load(open(filename, 'rb'))

y_pred = rf_load.predict(train)
print(y_pred)

