import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Create Dictionaries for Yes/No and Male/Female values
YesNoValues = {'Y': '1',
               'N': '0'}

GenderValues = {'M': '1',
                'F': '0'}

# Take inputs from the user

age = int(input("Enter your age : "))
gender = input("Enter your gender (M/F) : ").upper()
dl = input("Do you have a Driver's license? (Y/N) : ").upper()
region = input("Which region do you belong to? : ")
prevIns = input("Do you have a previous insurance? (Y/N) : ").upper()
vehicle_age = float(input("How old is your Vehicle? (No. of Years) : "))
vintage = int(input("Number of Days have you been associated with the company? (Vintage) : "))
premium = input("What amount are you currently paying as Insurance premium? : ")
vehicle_damage = input("Is the vehicle damaged? (Y/N) : ").upper()
policy_sales = input("Please enter the code for channel of outreaching to you: ")

# Check if all the inputs fall within the acceptable range of their allowed values
if (18 < age < 100) and (gender == 'M' or gender == 'F') and (
        dl == 'Y' or dl == 'N') and region.isdigit() and (
        prevIns == 'Y' or prevIns == 'N') and (premium.isdecimal()) and (
        vehicle_damage == 'Y' or vehicle_damage == 'N') and (policy_sales.isdigit()):

    # Change the values into categories since that was the training model
    gender = GenderValues.get(gender)
    dl = YesNoValues.get(dl)
    region = str(region)
    prevIns = YesNoValues.get(prevIns)
    vehicle_age_lt_1 = '0'
    vehicle_age_gt_2 = '0'
    if vehicle_age < 1.0:
        vehicle_age_lt_1 = '1'
    elif vehicle_age > 2.0:
        vehicle_age_gt_2 = '1'
    premium = str(premium)
    vehicle_damage_yes = YesNoValues.get(vehicle_damage)
    policy_sales = str(policy_sales)

    # Create a dictionary of user input based on these values
    user_input = {'Gender': [gender],
                  'Age': [age],
                  'Driving_License': [dl],
                  'Region_Code': [region],
                  'Previously_Insured': [prevIns],
                  'Annual_Premium': [premium],
                  'Policy_Sales_Channel': [policy_sales],
                  'Vintage': [vintage],
                  'Vehicle_Age_lt_1_Year': [vehicle_age_lt_1],
                  'Vehicle_Age_gt_2_Years': [vehicle_age_gt_2],
                  'Vehicle_Damage_Yes': [vehicle_damage_yes]
                  }

    # Convert the dictionary into a Dataframe so that we can apply prediction on it
    test = pd.DataFrame(data=user_input)

    # Data Prediction begins - fetch the saved model (Random Forest) in this case
    filename = 'rf_bal_model.sav'
    rf_load = pickle.load(open(filename, 'rb'))

    # Make the prediction now
    y_pred = rf_load.predict(test)
    print("\n")
    if y_pred == 0:
        print("You are not likely to buy Vehicle insurance!")
    else:
        print("I think you will buy a Vehicle insurance!")
else:
    print("Invalid input !!!")

# end of code \m/
