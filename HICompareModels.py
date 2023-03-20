import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, classification_report
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# Set background for the grid
sns.set_style('darkgrid', {"axes.facecolor": ".8"})

train = pd.read_csv('HI_Pred_Balanced_Train.csv')
test = pd.read_csv('HI_Pred_test.csv')

# Divide columns - numeric and categorical
num_feat = ['Age', 'Vintage']
cat_feat = ['Gender', 'Driving_License', 'Previously_Insured', 'Vehicle_Age_lt_1_Year', 'Vehicle_Age_gt_2_Years',
            'Vehicle_Damage_Yes', 'Region_Code', 'Policy_Sales_Channel']

# Preprocessing begins

# Gender can be made as categorical - Female = 0 and Male = 1
train['Gender'] = train['Gender'].map({'Female': 0, 'Male': 1}).astype(int)
test['Gender'] = test['Gender'].map({'Female': 0, 'Male': 1}).astype(int)

# Get categorical variables - Vehicle age and Vehicle Damage; drop the first category as it is redundant
train = pd.get_dummies(train, drop_first=True)
test = pd.get_dummies(test, drop_first=True)

# Rename column names containing special characters '</>' and set their type as integer
train = train.rename(
    columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
train['Vehicle_Age_lt_1_Year'] = train['Vehicle_Age_lt_1_Year'].astype('int')
train['Vehicle_Age_gt_2_Years'] = train['Vehicle_Age_gt_2_Years'].astype('int')
train['Vehicle_Damage_Yes'] = train['Vehicle_Damage_Yes'].astype('int')

test = test.rename(
    columns={"Vehicle_Age_< 1 Year": "Vehicle_Age_lt_1_Year", "Vehicle_Age_> 2 Years": "Vehicle_Age_gt_2_Years"})
test['Vehicle_Age_lt_1_Year'] = test['Vehicle_Age_lt_1_Year'].astype('int')
test['Vehicle_Age_gt_2_Years'] = test['Vehicle_Age_gt_2_Years'].astype('int')
test['Vehicle_Damage_Yes'] = test['Vehicle_Damage_Yes'].astype('int')

# Drop ID column as it is not required
train = train.drop('id', axis=1)
test = test.drop('id', axis=1)

# Store all categorical columns as String
for column in cat_feat:
    train[column] = train[column].astype('str')
    test[column] = test[column].astype('str')

# Create the Data Model
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold

# Get response values from Training data and then drop the column from the dataset
train_target = train['Response']
train = train.drop(['Response'], axis=1)

# Split the training data randomly into train and test subsets
x_train, x_test, y_train, y_test = train_test_split(train, train_target, random_state=0)

# First Model - Random Forest

# Criterion - Gini : Range [0.0.5] - Tells the purity (Pure node - linked to single class)
# Entropy : Range[0,1]
random_search = {'criterion': ['entropy', 'gini'],
                 'max_depth': [2, 3, 4, 5, 6, 7, 10],
                 'min_samples_leaf': [4, 6, 8],
                 'min_samples_split': [5, 7, 10],
                 'n_estimators': [300]}

clf = RandomForestClassifier()
model = RandomizedSearchCV(estimator=clf, param_distributions=random_search, n_iter=10,
                           cv=4, verbose=1, random_state=101, n_jobs=-1)
model.fit(x_train, y_train)

# Testing the model now
y_pred = model.predict(x_test)

# Print classification report
print("****** Random Forest *****")
print(classification_report(y_test, y_pred))

# Plot the ROC Curve
y_score = model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
print('Area under curve (AUC): ', auc(fpr, tpr))
print("\n")

# Save the model !!
filename = 'RandomForestModel.sav'
pickle.dump(model, open(filename, 'wb'))

# Second Model - Logistic Regression

model = LogisticRegression(max_iter=10000)
model.fit(x_train, y_train)

# Testing the model now
y_pred = model.predict(x_test)
# Print classification report
print("****** Logistic Regression *****")
print(classification_report(y_test, y_pred))

# Plot the ROC Curve
y_score = model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
print('Area under curve (AUC): ', auc(fpr, tpr))
print("\n")

# Save the model !!
filename = 'LogisticRegressionModel.sav'
pickle.dump(model, open(filename, 'wb'))

# Third Model - GradientBoostingClassifier

model = GradientBoostingClassifier()
model.fit(x_train, y_train)

# Testing the model now
y_pred = model.predict(x_test)
# Print classification report
print("****** GradientBoostingClassifier *****")
print(classification_report(y_test, y_pred))

# Plot the ROC Curve
y_score = model.predict_proba(x_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_score)
print('Area under curve (AUC): ', auc(fpr, tpr))
print("\n")

# Save the model !!
filename = 'GradientBoostingClassifier.sav'
pickle.dump(model, open(filename, 'wb'))