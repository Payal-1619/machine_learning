#!/usr/bin/env python
# coding: utf-8

# # Assignment - Decision Trees and Random Forests
# 
# ![](https://i.imgur.com/3sw1fY9.jpg)
# 
# In this assignment, you'll continue building on the previous assignment to predict the price of a house using information like its location, area, no. of rooms etc. You'll use the dataset from the [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) competition on [Kaggle](https://kaggle.com). 
# 
# We'll follow a step-by-step process:
# 
# 1. Download and prepare the dataset for training
# 2. Train, evaluate and interpret a decision tree
# 3. Train, evaluate and interpret a random forest
# 4. Tune hyperparameters to improve the model
# 5. Make predictions and save the model
# 
# As you go through this notebook, you will find a **???** in certain places. Your job is to replace the **???** with appropriate code or values, to ensure that the notebook runs properly end-to-end and your machine learning model is trained properly without errors. 
# 
# **Guidelines**
# 
# 1. Make sure to run all the code cells in order. Otherwise, you may get errors like `NameError` for undefined variables.
# 2. Do not change variable names, delete cells, or disturb other existing code. It may cause problems during evaluation.
# 3. In some cases, you may need to add some code cells or new statements before or after the line of code containing the **???**. 
# 4. Since you'll be using a temporary online service for code execution, save your work by running `jovian.commit` at regular intervals.
# 5. Review the "Evaluation Criteria" for the assignment carefully and make sure your submission meets all the criteria.
# 6. Questions marked **(Optional)** will not be considered for evaluation and can be skipped. They are for your learning.
# 7. It's okay to ask for help & discuss ideas on the [community forum](https://jovian.ai/forum/c/zero-to-gbms/gbms-assignment-2/99), but please don't post full working code, to give everyone an opportunity to solve the assignment on their own.
# 
# 
# **Important Links**:
# 
# - Make a submission here: https://jovian.ai/learn/machine-learning-with-python-zero-to-gbms/assignment/assignment-2-decision-trees-and-random-forests
# - Ask questions, discuss ideas and get help here: https://jovian.ai/forum/c/zero-to-gbms/gbms-assignment-2/99
# - Review this Jupyter notebook: https://jovian.ai/aakashns/sklearn-decision-trees-random-forests
# 

# ## How to Run the Code and Save Your Work
# 
# **Option 1: Running using free online resources (1-click, recommended):** The easiest way to start executing the code is to click the **Run** button at the top of this page and select **Run on Binder**. This will set up a cloud-based Jupyter notebook server and allow you to modify/execute the code.
# 
# 
# **Option 2: Running on your computer locally:** To run the code on your computer locally, you'll need to set up [Python](https://www.python.org), download the notebook and install the required libraries. Click the **Run** button at the top of this page, select the **Run Locally** option, and follow the instructions.
# 
# **Saving your work**: You can save a snapshot of the assignment to your [Jovian](https://jovian.ai) profile, so that you can access it later and continue your work. Keep saving your work by running `jovian.commit` from time to time.

# In[7]:


get_ipython().system('pip install jovian --upgrade --quiet')


# In[8]:


import jovian


# In[9]:


jovian.commit(project='python-random-forests-assignment', privacy='secret')


# Let's begin by installing the required libraries.

# In[10]:


get_ipython().system('pip install opendatasets scikit-learn plotly folium --upgrade --quiet')


# In[11]:


get_ipython().system('pip install pandas numpy matplotlib seaborn --quiet')


# ## Download and prepare the dataset for training

# In[12]:


import os
from zipfile import ZipFile
from urllib.request import urlretrieve

dataset_url = 'https://github.com/JovianML/opendatasets/raw/master/data/house-prices-advanced-regression-techniques.zip'
urlretrieve(dataset_url, 'house-prices.zip')
with ZipFile('house-prices.zip') as f:
    f.extractall(path='house-prices')
    
os.listdir('house-prices')


# In[13]:


import pandas as pd
pd.options.display.max_columns = 200
pd.options.display.max_rows = 200

prices_df = pd.read_csv('house-prices/train.csv')
prices_df


# In[14]:


import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split

# Identify input and target columns
input_cols, target_col = prices_df.columns[1:-1], prices_df.columns[-1]
inputs_df, targets = prices_df[input_cols].copy(), prices_df[target_col].copy()

# Identify numeric and categorical columns
numeric_cols = prices_df[input_cols].select_dtypes(include=np.number).columns.tolist()
categorical_cols = prices_df[input_cols].select_dtypes(include='object').columns.tolist()

# Impute and scale numeric columns
imputer = SimpleImputer().fit(inputs_df[numeric_cols])
inputs_df[numeric_cols] = imputer.transform(inputs_df[numeric_cols])
scaler = MinMaxScaler().fit(inputs_df[numeric_cols])
inputs_df[numeric_cols] = scaler.transform(inputs_df[numeric_cols])

# One-hot encode categorical columns
encoder = OneHotEncoder(sparse=False, handle_unknown='ignore').fit(inputs_df[categorical_cols])
encoded_cols = list(encoder.get_feature_names(categorical_cols))
inputs_df[encoded_cols] = encoder.transform(inputs_df[categorical_cols])

# Create training and validation sets
train_inputs, val_inputs, train_targets, val_targets = train_test_split(
    inputs_df[numeric_cols + encoded_cols], targets, test_size=0.25, random_state=42)


# Let's save our work before continuing.

# In[16]:


jovian.commit()


# ## Decision Tree
# 

# > **QUESTION 1**: Train a decision tree regressor using the training set.

# In[17]:


from sklearn.tree import DecisionTreeRegressor


# In[18]:


# Create the model
tree = DecisionTreeRegressor(random_state=42)


# In[19]:


# Fit the model to the training data
tree.fit(train_inputs,train_targets)


# Let's save our work before continuing.

# In[20]:


jovian.commit()


# > **QUESTION 2**: Generate predictions on the training and validation sets using the trained decision tree, and compute the RMSE loss.

# In[21]:


from sklearn.metrics import mean_squared_error


# In[22]:


tree_train_preds = tree.predict(train_inputs[numeric_cols + encoded_cols])


# In[23]:


tree_train_rmse = mean_squared_error(train_targets,tree_train_preds, squared=False)


# In[24]:


tree_val_preds = tree.predict(val_inputs[numeric_cols + encoded_cols])


# In[25]:


tree_val_rmse = mean_squared_error(val_targets,tree_val_preds, squared=False)


# In[26]:


print('Train RMSE: {}, Validation RMSE: {}'.format(tree_train_rmse, tree_val_rmse))


# In[27]:


from sklearn.metrics import accuracy_score
accuracy_score(train_targets,tree_train_preds)


# In[ ]:





# Let's save our work before continuing.

# In[28]:


jovian.commit()


# > **QUESTION 3**: Visualize the decision tree (graphically and textually) and display feature importances as a graph. Limit the maximum depth of graphical visualization to 3 levels.

# In[29]:


import matplotlib.pyplot as plt
from sklearn.tree import plot_tree, export_text
import seaborn as sns
sns.set_style('darkgrid')
get_ipython().run_line_magic('matplotlib', 'inline')


# In[30]:


plt.figure(figsize=(30,15))

# Visualize the tree graphically using plot_tree
p=plot_tree(tree, feature_names=train_inputs.columns, max_depth=3, filled=True)


# In[31]:


# Visualize the tree textually using export_text
tree_text = export_text(tree,max_depth=10,feature_names=list(train_inputs.columns))


# In[32]:


# Display the first few lines
print(tree_text[:2000])


# In[33]:


# Check feature importance
tree_importances = tree.feature_importances_


# In[34]:


tree_importance_df = pd.DataFrame({
    'feature': train_inputs.columns,
    'importance': tree_importances
}).sort_values('importance', ascending=False)


# In[35]:


tree_importance_df


# In[168]:


plt.title('Decision Tree Feature Importance')
sns.barplot(data=tree_importance_df.head(10), x='importance', y='feature');


# In[ ]:





# Let's save our work before continuing.

# In[37]:


jovian.commit()


# ## Random Forests
# 

# > **QUESTION 4**: Train a random forest regressor using the training set.

# In[38]:


from sklearn.ensemble import RandomForestRegressor


# In[39]:


# Create the model
rf1 = RandomForestRegressor(n_jobs=-1,random_state=42)


# In[40]:


# Fit the model
rf1.fit(train_inputs[numeric_cols + encoded_cols],train_targets)


# In[ ]:





# Let's save our work before continuing.

# In[41]:


jovian.commit()


# > **QUESTION 5**: Make predictions using the random forest regressor.

# In[42]:


rf1_train_preds = rf1.predict(train_inputs[numeric_cols + encoded_cols])


# In[43]:


rf1_train_rmse = mean_squared_error(train_targets,rf1_train_preds,squared=False)


# In[44]:


rf1_val_preds = rf1.predict(val_inputs[numeric_cols + encoded_cols])


# In[45]:


rf1_val_rmse = mean_squared_error(val_targets,rf1_val_preds,squared=False)


# In[46]:


print('Train RMSE: {}, Validation RMSE: {}'.format(rf1_train_rmse, rf1_val_rmse))


# In[ ]:





# Let's save our work before continuing.

# In[47]:


jovian.commit()


# ## Hyperparameter Tuning
# 
# Let us now tune the hyperparameters of our model. You can find the hyperparameters for `RandomForestRegressor` here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
# 
# <img src="https://i.imgur.com/EJCrSZw.png" width="480">
# 
# Hyperparameters are use

# Let's define a helper function `test_params` which can test the given value of one or more hyperparameters.

# In[105]:


def test_params(**params):
    model = RandomForestRegressor(random_state=42, n_jobs=-1, **params).fit(train_inputs, train_targets)
    train_rmse = mean_squared_error(model.predict(train_inputs), train_targets, squared=False)
    val_rmse = mean_squared_error(model.predict(val_inputs), val_targets, squared=False)
    return train_rmse, val_rmse


# It can be used as follows:

# In[106]:


test_params(n_estimators=20, max_depth=20)


# In[107]:


test_params(n_estimators=50, max_depth=10, min_samples_leaf=4, max_features=0.4)


# Let's also define a helper function to test and plot different values of a single parameter.

# In[108]:


def test_param_and_plot(param_name, param_values):
    train_errors, val_errors = [], [] 
    for value in param_values:
        params = {param_name: value}
        train_rmse, val_rmse = test_params(**params)
        train_errors.append(train_rmse)
        val_errors.append(val_rmse)
    plt.figure(figsize=(10,6))
    plt.title('Overfitting curve: ' + param_name)
    plt.plot(param_values, train_errors, 'b-o')
    plt.plot(param_values, val_errors, 'r-o')
    plt.xlabel(param_name)
    plt.ylabel('RMSE')
    plt.legend(['Training', 'Validation'])


# In[109]:


test_param_and_plot('max_depth', [5, 10, 15, 20, 25, 30, 35])


# From the above graph, it appears that the best value for `max_depth` is around 20, beyond which the model starts to overfit.

# In[ ]:





# Let's save our work before continuing.

# In[110]:


jovian.commit()


# > **QUESTION 6**: Use the `test_params` and `test_param_and_plot` functions to experiment with different values of the  hyperparmeters like `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`, `min_weight_fraction_leaf`, `max_features`, `max_leaf_nodes`, `min_impurity_decrease`, `min_impurity_split` etc. You can learn more about the hyperparameters here: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

# In[148]:


test_params(n_estimators=500, max_depth=15, max_leaf_nodes=8, min_samples_leaf=4)


# In[149]:


test_params(n_estimators=500, max_depth=20,max_features=0.5, min_samples_leaf=4,min_samples_split=2,min_impurity_decrease=0.2)


# In[150]:


test_param_and_plot('max_features', [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1])


# In[145]:


test_param_and_plot('min_samples_leaf', [2,4,6,8,10,12])


# Let's save our work before continuing.

# In[151]:


jovian.commit()


# ## Training the Best Model
# 
# > **QUESTION 7**: Train a random forest regressor model with your best hyperparameters to minimize the validation loss.

# In[152]:


# Create the model with custom hyperparameters
rf2 = RandomForestRegressor(n_jobs=-1,random_state=42,
                            n_estimators=500,max_depth=20,
                            min_samples_leaf=4, max_features=0.5, min_samples_split=12 )


# In[153]:


# Train the model
rf2.fit(train_inputs,train_targets)


# In[ ]:





# Let's save our work before continuing.

# In[154]:


jovian.commit()


# > **QUESTION 8**: Make predictions and evaluate your final model. If you're unhappy with the results, modify the hyperparameters above and try again.

# In[155]:


rf2_train_preds = rf2.predict(train_inputs[numeric_cols + encoded_cols])


# In[156]:


rf2_train_rmse = mean_squared_error(train_targets,rf2_train_preds,squared=False)


# In[157]:


rf2_val_preds = rf2.predict(val_inputs[numeric_cols + encoded_cols])


# In[158]:


rf2_val_rmse = mean_squared_error(val_targets,rf1_val_preds,squared=False)


# In[159]:


print('Train RMSE: {}, Validation RMSE: {}'.format(rf2_train_rmse, rf2_val_rmse))


# In[ ]:





# Let's also view and plot the feature importances.

# In[160]:


rf2_importance_df = pd.DataFrame({
    'feature': train_inputs.columns,
    'importance': rf2.feature_importances_
}).sort_values('importance', ascending=False)


# In[161]:


rf2_importance_df


# In[170]:


plt.title('Decision Tree Feature Importance')
sns.barplot(data=rf2_importance_df.head(15), x='importance', y='feature');


# In[ ]:





# Let's save our work before continuing.

# In[ ]:


jovian.commit()


# ## Make a Submission
# 
# To make a submission, just execute the following cell:

# In[ ]:


jovian.submit('zerotogbms-a2')


# You can also submit your Jovian notebook link on the assignment page: https://jovian.ai/learn/machine-learning-with-python-zero-to-gbms/assignment/assignment-2-decision-trees-and-random-forests
# 
# Make sure to review the evaluation criteria carefully. You can make any number of submissions, and only your final submission will be evalauted.
# 
# Ask questions, discuss ideas and get help here: https://jovian.ai/forum/c/zero-to-gbms/gbms-assignment-2/99
# 
# NOTE: **The rest of this assignment is optional.**

# ## Making Predictions on the Test Set
# 
# Let's make predictions on the test set provided with the data.

# In[ ]:


test_df = pd.read_csv('house-prices/test.csv')


# In[ ]:


test_df


# First, we need to reapply all the preprocessing steps.

# In[ ]:


test_df[numeric_cols] = imputer.transform(test_df[numeric_cols])
test_df[numeric_cols] = scaler.transform(test_df[numeric_cols])
test_df[encoded_cols] = encoder.transform(test_df[categorical_cols])


# In[ ]:


test_inputs = test_df[numeric_cols + encoded_cols]


# We can now make predictions using our final model.

# In[ ]:


test_preds = rf2.predict(test_inputs)


# In[ ]:


submission_df = pd.read_csv('house-prices/sample_submission.csv')


# In[ ]:


submission_df


# Let's replace the values of the `SalePrice` column with our predictions.

# In[ ]:


submission_df['SalePrice'] = test_preds


# Let's save it as a CSV file and download it.

# In[ ]:


submission_df.to_csv('submission.csv', index=False)


# In[ ]:


from IPython.display import FileLink
FileLink('submission.csv') # Doesn't work on Colab, use the file browser instead to download the file.


# We can now submit this file to the competition: https://www.kaggle.com/c/house-prices-advanced-regression-techniques/submissions
# 
# ![](https://i.imgur.com/6h2vXRq.png)
# 

# > **(OPTIONAL) QUESTION**: Submit your predictions to the competition. Experiment with different models, feature engineering strategies and hyperparameters and try to reach the top 10% on the leaderboard.

# In[ ]:





# In[ ]:





# Let's save our work before continuing.

# In[6]:


jovian.commit()


# ### Making Predictions on Single Inputs

# In[ ]:


def predict_input(model, single_input):
    input_df = pd.DataFrame([single_input])
    input_df[numeric_cols] = imputer.transform(input_df[numeric_cols])
    input_df[numeric_cols] = scaler.transform(input_df[numeric_cols])
    input_df[encoded_cols] = encoder.transform(input_df[categorical_cols].values)
    return model.predict(input_df[numeric_cols + encoded_cols])[0]


# In[ ]:


sample_input = { 'MSSubClass': 20, 'MSZoning': 'RL', 'LotFrontage': 77.0, 'LotArea': 9320,
 'Street': 'Pave', 'Alley': None, 'LotShape': 'IR1', 'LandContour': 'Lvl', 'Utilities': 'AllPub',
 'LotConfig': 'Inside', 'LandSlope': 'Gtl', 'Neighborhood': 'NAmes', 'Condition1': 'Norm', 'Condition2': 'Norm',
 'BldgType': '1Fam', 'HouseStyle': '1Story', 'OverallQual': 4, 'OverallCond': 5, 'YearBuilt': 1959,
 'YearRemodAdd': 1959, 'RoofStyle': 'Gable', 'RoofMatl': 'CompShg', 'Exterior1st': 'Plywood',
 'Exterior2nd': 'Plywood', 'MasVnrType': 'None','MasVnrArea': 0.0,'ExterQual': 'TA','ExterCond': 'TA',
 'Foundation': 'CBlock','BsmtQual': 'TA','BsmtCond': 'TA','BsmtExposure': 'No','BsmtFinType1': 'ALQ',
 'BsmtFinSF1': 569,'BsmtFinType2': 'Unf','BsmtFinSF2': 0,'BsmtUnfSF': 381,
 'TotalBsmtSF': 950,'Heating': 'GasA','HeatingQC': 'Fa','CentralAir': 'Y','Electrical': 'SBrkr', '1stFlrSF': 1225,
 '2ndFlrSF': 0, 'LowQualFinSF': 0, 'GrLivArea': 1225, 'BsmtFullBath': 1, 'BsmtHalfBath': 0, 'FullBath': 1,
 'HalfBath': 1, 'BedroomAbvGr': 3, 'KitchenAbvGr': 1,'KitchenQual': 'TA','TotRmsAbvGrd': 6,'Functional': 'Typ',
 'Fireplaces': 0,'FireplaceQu': np.nan,'GarageType': np.nan,'GarageYrBlt': np.nan,'GarageFinish': np.nan,'GarageCars': 0,
 'GarageArea': 0,'GarageQual': np.nan,'GarageCond': np.nan,'PavedDrive': 'Y', 'WoodDeckSF': 352, 'OpenPorchSF': 0,
 'EnclosedPorch': 0,'3SsnPorch': 0, 'ScreenPorch': 0, 'PoolArea': 0, 'PoolQC': np.nan, 'Fence': np.nan, 'MiscFeature': 'Shed',
 'MiscVal': 400, 'MoSold': 1, 'YrSold': 2010, 'SaleType': 'WD', 'SaleCondition': 'Normal'}


# In[ ]:


predicted_price = predict_input(rf2, sample_input)


# In[ ]:


print('The predicted sale price of the house is ${}'.format(predicted_price))


# > **EXERCISE**: Change the sample input above and make predictions. Try different examples and try to figure out which columns have a big impact on the sale price. Hint: Look at the feature importance to decide which columns to try.

# In[ ]:





# In[ ]:





# ### Saving the Model

# In[ ]:


import joblib


# In[ ]:


house_prices_rf = {
    'model': rf2,
    'imputer': imputer,
    'scaler': scaler,
    'encoder': encoder,
    'input_cols': input_cols,
    'target_col': target_col,
    'numeric_cols': numeric_cols,
    'categorical_cols': categorical_cols,
    'encoded_cols': encoded_cols
}


# In[ ]:


joblib.dump(house_prices_rf, 'house_prices_rf.joblib')


# Let's save our work before continuing.

# In[ ]:


jovian.commit(outputs=['house_prices_rf.joblib'])


# In[ ]:





# ### Predicting the Logarithm of Sale Price

# > **(OPTIONAL) QUESTION**: In the [original Kaggle competition](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview/evaluation), the model is evaluated by computing the Root Mean Squared Error on the logarithm of the sale price. Try training a random forest to predict the logarithm of the sale price, instead of the actual sales price and see if the results you obtain are better than the models trained above.

# In[ ]:





# In[ ]:




