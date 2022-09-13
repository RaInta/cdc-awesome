# Databricks notebook source
# MAGIC %md
# MAGIC # Interpretable Machine Learning
# MAGIC 
# MAGIC _&copy; 2022 Ra Inta, for BH Analytics_
# MAGIC 
# MAGIC This notebook provides the code used to get the slides for the accompanying seminar _Interpretable Machine Learning_ (seminar 14 in the 2022 CDC Advanced Data Science Seminar series).

# COMMAND ----------

# MAGIC %md
# MAGIC Import necessary packages

# COMMAND ----------

import numpy as np
import pandas as pd

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns

from patsy import dmatrices

import statsmodels as sm
import statsmodels.formula.api as smf

import sklearn
from sklearn.metrics import confusion_matrix

# COMMAND ----------

# MAGIC %md
# MAGIC Make an `images` directory if if doesn't exist:

# COMMAND ----------

import os

if not os.path.isdir('images'):
    os.mkdir('images')

# COMMAND ----------

# MAGIC %md
# MAGIC Print out module versions

# COMMAND ----------

def print_module_version(module):
    print(f"{module.__name__} version: {module.__version__}")

# COMMAND ----------

print_module_version(np)
print_module_version(pd)
print_module_version(mpl)
print_module_version(sns)
print_module_version(sklearn)
print_module_version(sm)

# COMMAND ----------

# MAGIC %md
# MAGIC Set some plotting parameters:

# COMMAND ----------

# MAGIC %matplotlib inline
# MAGIC plt.rcParams['figure.figsize'] = [10.0, 6.0]
# MAGIC plt.rcParams['font.serif'] = "Georgia"
# MAGIC plt.rcParams['font.family'] = "serif"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Regression: Health Insurance Charges
# MAGIC 
# MAGIC This dataset is an examination of health insurance charges in the US. Each row (1338 in total) is a patient over the collection period. The primary outcome is `charges`, the amount each patient (customer) is charged for health insurance during this period. The other variables are: Age, Sex, BMI, Number of Children, Smoker and Region of the US. 
# MAGIC 
# MAGIC There are no missing or undefined values in the dataset.
# MAGIC 
# MAGIC You may download this dataset from Kaggle:
# MAGIC 
# MAGIC https://www.kaggle.com/datasets/mirichoi0218/insurance

# COMMAND ----------

insurance = pd.read_csv('health_data/insurance.csv')

# COMMAND ----------

insurance.shape

# COMMAND ----------

insurance.isnull().mean()

# COMMAND ----------

# MAGIC %md
# MAGIC Rename the columns so they read better on a plot:

# COMMAND ----------

insurance.rename(columns={x: x.title() for x in insurance.columns}, inplace=True)

insurance.rename(columns={'Bmi': 'BMI'}, inplace=True)

# COMMAND ----------

# MAGIC %md
# MAGIC Examine the first five rows:

# COMMAND ----------

insurance.round(2).head()

# COMMAND ----------

# MAGIC %md
# MAGIC Perform a log-transform of the charges data:

# COMMAND ----------

insurance["Log(Charges)"] = np.log(insurance["Charges"] + 0.01)

# COMMAND ----------

sns.set_context("talk")

# COMMAND ----------

sns.displot(x="Charges", data=insurance, kde=True)
plt.savefig("images/insurance_charges_kde.png")

# COMMAND ----------

sns.displot(x="Log(Charges)", data=insurance, kde=True)
plt.savefig("images/insurance_log_charges_kde.png")

# COMMAND ----------

# MAGIC %md
# MAGIC Teaser: plot the regression as a function of `Age`:

# COMMAND ----------

sns.lmplot(x="Age", y="Charges", line_kws={"color": "firebrick"}, data=insurance)

# Add annotations to the plot
plt.text(x=20, y=55000,
         s=r"$\hat{y} = \$3,166 + \$258/yr\times Age$",
         fontsize=18,
         bbox=dict(facecolor='white', alpha=1),
         color="firebrick")

plt.savefig("images/insurance_age_regression.png")

# COMMAND ----------

reg_age = smf.ols(formula="Charges ~ Age + Smoker + Region", data=insurance).fit()

reg_age.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC Compare to the log-transformed version:

# COMMAND ----------

reg_log_age = smf.ols(formula="Q('Log(Charges)') ~ Age + Smoker + Region", data=insurance).fit()

reg_log_age.summary()

# COMMAND ----------

sns.lmplot(x="Age", y="Log(Charges)", hue="Smoker", data=insurance)

# COMMAND ----------

# MAGIC %md
# MAGIC Examine interaction effects between `BMI` and `Smoker`, plus the effect of `Age`, on the outcome `Charges`:

# COMMAND ----------

reg_bmi_smoker = smf.ols(formula="Charges ~ BMI:Smoker + Age", data=insurance).fit()

reg_bmi_smoker.summary()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Perform Machine Learning Regression

# COMMAND ----------

formula = "Charges ~ Age + Smoker + Region"

formula

# COMMAND ----------

## use Patsy to create model matrices
Y, X = dmatrices(formula, insurance)

# COMMAND ----------

## Split Data into training and sample
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    Y,
                                                    test_size=0.25,
                                                    random_state=1729)

# COMMAND ----------

# MAGIC %md
# MAGIC ### For Kitchen-sink Models

# COMMAND ----------



# COMMAND ----------



# COMMAND ----------

formula_all = "Charges ~ " + " + ".join(insurance.columns[:6]) + " + BMI:Smoker"

formula_all

# COMMAND ----------

## use Patsy to create model matrices
Y_all, X_all = dmatrices(formula_all, insurance)

# COMMAND ----------

## Split Data into training and sample
from sklearn.model_selection import train_test_split

X_all_train, X_all_test, y_all_train, y_all_test = train_test_split(X_all,
                                                    Y_all,
                                                    test_size=0.25,
                                                    random_state=1729)

# COMMAND ----------

## import linear model
from sklearn import linear_model

## Define model parameters
reg = linear_model.LinearRegression()

## fit model using data with .fit
reg.fit(X_train, y_train)

# COMMAND ----------

reg.score(X_train, y_train)

# COMMAND ----------

reg.score(X_test, y_test)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Apply some regularization:

# COMMAND ----------

lasso = linear_model.LassoCV(alphas=[0.01, 0.1, 1, 10, 1000, 10000]).fit(X_train, y_train.ravel())

# COMMAND ----------

lasso.score(X_train, y_train)

# COMMAND ----------

lasso.score(X_test, y_test)

# COMMAND ----------

lasso.alpha_

# COMMAND ----------

reg_full = linear_model.LinearRegression().fit(X, Y)

## fit model using data with .fit
reg.score(X, Y)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Tree-based regression model

# COMMAND ----------

from sklearn.tree import DecisionTreeRegressor, plot_tree

decision_tree = DecisionTreeRegressor(random_state=1729)

decision_tree.name = "Decision Tree"

decision_tree.fit(X_train, y_train)

# COMMAND ----------

plt.figure(figsize=(12,12))

g = plot_tree(decision_tree, 
               fontsize=24, 
               feature_names=X.design_info.column_names,
               max_depth=2)

plt.savefig('images/decision_tree.png')

# COMMAND ----------

from itertools import product

age_ranges = np.linspace(18, 65, 100)

smoker = ["yes", "no"]

df_predict = pd.DataFrame(product(age_ranges, smoker), columns=["Age", "Smoker"])

df_predict["Region"] = "northeast"

df_predict["Charges"] = 0  # This will be dropped immediately

# Kludge to get all region levels
for region in ["northwest", "southeast", "southwest"]:
    df_predict = df_predict.append(df_predict.iloc[-1, :])
    df_predict.iat[-1, -2] = region

# COMMAND ----------

Y_predict, X_predict = dmatrices("Charges ~ Age + Smoker + Region", df_predict)

# COMMAND ----------

df_predict["Charges (predicted)"] = decision_tree.predict(X_predict)

# COMMAND ----------

ax = sns.scatterplot(x="Age", hue="Smoker", y="Charges", data=insurance, alpha=0.5)


sns.lineplot(x="Age", y="Charges (predicted)", hue="Smoker", data=df_predict, ax=ax)
plt.title("Decision Tree Regression")

plt.savefig("images/decision_tree_lineplot.png")

# COMMAND ----------

decision_tree_less_depth = DecisionTreeRegressor(random_state=1729, max_depth=4)

decision_tree_less_depth.name = "Decision Tree (less depth)"

decision_tree_less_depth.fit(X_train, y_train)

# COMMAND ----------

plt.figure(figsize=(12,12))

g = plot_tree(decision_tree_less_depth, 
               fontsize=24, 
               feature_names=X.design_info.column_names,
               max_depth=2)

plt.savefig('images/decision_tree_less_depth.png')

# COMMAND ----------

df_predict["Charges (predicted; less depth)"] = decision_tree_less_depth.predict(X_predict)

# COMMAND ----------

ax = sns.scatterplot(x="Age", hue="Smoker", y="Charges", data=insurance, alpha=0.5)

sns.lineplot(x="Age", y="Charges (predicted; less depth)", hue="Smoker", data=df_predict, ax=ax)
plt.title("Decision Tree Regression (max depth: 4)")

plt.savefig("images/decision_tree_less_depth_lineplot.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Partial Dependence Plots (PDPs)

# COMMAND ----------

from sklearn.inspection import PartialDependenceDisplay

# Examine PDPs for Smoker and Age
# PartialDependenceDisplay.from_estimator(decision_tree, X, [1, 5, (1, 5)],
#                                         feature_names=["Smoker", "Age", ("Smoker", "Age")])
                                        
PartialDependenceDisplay.from_estimator(decision_tree, X, [1, 5, (1, 5)],
                                        feature_names=X.design_info.column_names)

plt.savefig("images/pdp_age_smoker.png")

# COMMAND ----------

from sklearn.inspection import PartialDependenceDisplay

# Examine PDPs for Smoker and Age
# PartialDependenceDisplay.from_estimator(decision_tree, X, [1, 5, (1, 5)],
#                                         feature_names=["Smoker", "Age", ("Smoker", "Age")])
                                        
PartialDependenceDisplay.from_estimator(decision_tree, X, [5],
                                        feature_names=X.design_info.column_names, kind="individual")

plt.savefig("images/ice_age_smoker_decisionTree.png")

# COMMAND ----------

from sklearn.inspection import PartialDependenceDisplay

# Examine PDPs for Smoker and Age
# PartialDependenceDisplay.from_estimator(decision_tree, X, [1, 5, (1, 5)],
#                                         feature_names=["Smoker", "Age", ("Smoker", "Age")])
                                        
PartialDependenceDisplay.from_estimator(reg, X, [5],
                                        feature_names=X.design_info.column_names, kind="individual")

plt.savefig("images/ice_age_smoker_linearRegression.png")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Permutation Feature Importance

# COMMAND ----------

decision_tree_all = DecisionTreeRegressor(random_state=1729)

decision_tree_all.name = "Decision Tree (all)"

decision_tree_all.fit(X_all_train, y_all_train)

# COMMAND ----------

import eli5
from eli5.sklearn import PermutationImportance

perm = PermutationImportance(decision_tree_all, random_state=1729).fit(X_all_train, y_all_train)
eli5.show_weights(perm, feature_names=X_all.design_info.column_names)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Global Surrogate Models

# COMMAND ----------

from xgboost import XGBRegressor

xgb_reg_all = XGBRegressor(random_state=1729)

# COMMAND ----------

xgb_reg_all.fit(X_all_train, y_all_train)

# COMMAND ----------

xgb_reg_all.score(X_all_train, y_all_train)

# COMMAND ----------

xgb_reg_all.score(X_all_test, y_all_test)

# COMMAND ----------

xgb_reg_all = XGBRegressor(random_state=1729)

# COMMAND ----------

# MAGIC %md
# MAGIC Sneaky trick to tell `scikit-learn` the XGBoost model was fitted:

# COMMAND ----------

xgb_reg_all.fitted_ = True

# COMMAND ----------

PartialDependenceDisplay.from_estimator(xgb_reg_all, X_all, [6, 7, (6, 7)],
                                        feature_names=X_all.design_info.column_names)

plt.savefig("images/pdp_age_smoker_xgb.png")

# COMMAND ----------

insurance["predict_xgb"] = xgb_reg_all.predict(X_all)

# COMMAND ----------

formula_all

# COMMAND ----------

global_surrogate = smf.ols(
    formula="predict_xgb ~ Age + Sex + BMI + Children + Smoker + Region + BMI:Smoker",
    data=insurance).fit()

global_surrogate.summary()


# COMMAND ----------

# MAGIC %md
# MAGIC ### LIME

# COMMAND ----------

import lime
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(X_all_train, 
feature_names=X_all.design_info.column_names, 
class_names=['Charges'], 
# categorical_features=categorical_features, 
verbose=True, mode='regression')


# COMMAND ----------

i = 25
exp = explainer.explain_instance(X_all_test[i], xgb_reg_all.predict, num_features=5)

# COMMAND ----------

exp.show_in_notebook(show_table=True)

# COMMAND ----------

exp.as_list()

# COMMAND ----------

with open("images/LIME.html", 'w', encoding="utf-8") as f:
    f.write(exp.as_html())

# COMMAND ----------

# MAGIC %md
# MAGIC ### SHAP

# COMMAND ----------

import shap

explainer = shap.Explainer(xgb_reg_all)

shap_values = explainer(X_all)




# COMMAND ----------

from shap.plots import _waterfall
_waterfall.waterfall_legacy(explainer.expected_value, 
shap_values[0].values, 
X_all[:, 0], 
feature_names=X_all.design_info.column_names)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
