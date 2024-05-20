#!/usr/bin/env python
# coding: utf-8

# # Global Food Analytics: Empowering Consumers and Better Understanding of Food Nutrition
# 
# The project aims to analyze global food product data for nutritional quality across dimensions (such as country, organic vs non organic) and health insights, utilizing the Open Food Facts database.
# 
# The goal is to develop a comprehensive predictive and exploratory model that aids consumers in making informed decisions regarding food nutrition and safety.
# 
# To explore a variety of ML models, including regression and classification models. Advanced techniques such as neural networks and deep learning may also be considered for complex pattern recognition tasks.

# ## Import the required Libraries

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from sklearn.model_selection import train_test_split
import torch
import torchvision
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from collections import Counter
from PIL import Image
from skimage import io, transform
import os
from torchvision.io import read_image
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from google.colab import drive
import json
import glob


# In[ ]:


get_ipython().run_cell_magic('capture', '', '!apt update\n!pip install kaggle')


# # **Part I:** Data Loading and Preprocessing

# ## **1.1** Data Loading and Preprocessing
# 
# We are using the csv file from Open Food Facts database. (https://www.kaggle.com/datasets/openfoodfacts/world-food-facts/data)
# 
# To get the data in here:
# 1. Go to this [Kaggle link](https://www.kaggle.com) and create a Kaggle account (unless you already have one)
# 2. Go to Account and click on "Create New API Token" to get the API key in the form of a json file `kaggle.json`
# 3. Upload the `kaggle.json` file to the default location in your Google Drive, 'MyDrive' (Please **DO NOT** upload the json file into any _specific_ folder!).
# 

# ### **1.1.1** Read Data

# In[ ]:


from google.colab import drive
drive.mount('/content/drive')


# In[ ]:


# Create the kaggle directory and
# (NOTE: Do NOT run this cell more than once unless restarting kernel)
get_ipython().system('mkdir ~/.kaggle')


# In[ ]:



# Read the uploaded kaggle.json file
get_ipython().system('cp /content/drive/MyDrive/kaggle.json ~/.kaggle/')


# In[ ]:


# Download dataset
get_ipython().getoutput('kaggle datasets download -d openfoodfacts/world-food-facts')


# In[ ]:


# Unzip folder in Colab content folder
get_ipython().system('unzip /content/world-food-facts.zip')


# In[ ]:


# Read the csv file and save it to a dataframe called "df_foodfacts"
df_foodfacts = pd.read_csv('en.openfoodfacts.org.products.tsv', sep='\t')


# Review the first 5 rows

# In[ ]:


df_foodfacts.head()


# ### **1.1.2** Check Nulls and Duplicates
# 
# We will find the number of rows with null values and the number of duplicated rows.
# 
# Store the results into `num_nulls` and `num_dups`, respectively.
# 

# In[ ]:


# find number of rows with null values
num_nulls = df_foodfacts.isnull().any(axis = 1).sum()
print(num_nulls)
print(len(df_foodfacts))


# The number of total rows is equal to the number or rows with null values. It implies that all rows have null values in certain columns. Then calculate the number of rows with null values for each column.

# In[ ]:


# calculate the number of nulls in each column
pd.options.display.max_rows=200 # increase max rows displayed

null_counts = df_foodfacts.isnull().sum().sort_values(ascending=True)
print(null_counts)


# As a next step, remove the columns with all rows with null values. We dropped 16 columns.

# In[ ]:


# Remove columns where all values are null
df_foodfacts = df_foodfacts.dropna(axis=1, how='all')

# Print the shape of the new DataFrame to see the result
print("New DataFrame shape:", df_foodfacts.shape)


# In[ ]:


# find number of duplicated rows
num_dups = df_foodfacts.duplicated().sum()
print(num_dups)


# 
# 
# ### **1.1.3** Data Cleaning

# Visualize the columns with missing / null values

# In[ ]:


def msv1(data, thresh=20, color_above='#ffcccb', color_below='#90ee90', width=18, height=10, bar_width=0.75):
    plt.figure(figsize=(width, height))
    percentage = (data.isnull().mean()) * 100
    sorted_series = percentage.sort_values(ascending=False)
    above_thresh = sorted_series > thresh
    colors = np.where(above_thresh, color_above, color_below)

    plt.text(len(sorted_series) - 1, thresh + 5, 'Threshold for missing values (%s%%)' % thresh, fontsize=16, color='red', ha='right')
    sorted_series.plot.bar(color=colors, width=bar_width, edgecolor=None)
    plt.axhline(y=thresh, color='r', linestyle='-')
    plt.title('Missing Values Percentage per Column', fontsize=20 )

    plt.xlabel('Columns', size=15)
    plt.ylabel('Missing Values Percentage', size=15)
    plt.xticks(rotation=90, weight='normal')
    plt.yticks(weight='normal')
    plt.tight_layout()

    return plt.show()


# In[ ]:


# Call the function with the desired threshold and colors
msv1(df_foodfacts, 75, color_above='#ffcccb', color_below='#90ee90')


# Excluding all missing columns with more than 75% missing values to avoid misleading results

# In[ ]:


df_foodfacts_cleaned=df_foodfacts.dropna(thresh=0.25*len(df_foodfacts), axis=1)
print(f"Data shape before cleaning {df_foodfacts.shape}")
print(f"Data shape after cleaning {df_foodfacts_cleaned.shape}")
print(f"We dropped {df_foodfacts.shape[1]- df_foodfacts_cleaned.shape[1]} columns")


# In[ ]:


#remove columns we are not interested in (like code, url, etc)
df_foodfacts_cleaned = df_foodfacts_cleaned[['product_name', 'packaging_tags', 'brands_tags', 'categories_en', 'countries_en', 'ingredients_text',
                                             'serving_size', 'additives_en', 'ingredients_from_palm_oil_n',
                                             'ingredients_that_may_be_from_palm_oil_n', 'nutrition_grade_fr', 'pnns_groups_1', 'pnns_groups_2',
                                             'states_en', 'main_category_en', 'energy_100g', 'fat_100g', 'saturated-fat_100g',
                                             'trans-fat_100g', 'cholesterol_100g', 'carbohydrates_100g', 'sugars_100g', 'fiber_100g', 'proteins_100g', 'salt_100g',
                                             'sodium_100g', 'vitamin-a_100g', 'vitamin-c_100g', 'calcium_100g', 'iron_100g', 'nutrition-score-fr_100g']]
print(f"Data shape after cleaning {df_foodfacts_cleaned.shape}")


# # **Part II:** EDA & Feature Engineering
# 
# We use Exploratory Data Analysis (EDA) approach to analyzing data sets to summarize their main characteristics. A statistical model can be used or not, but primarily EDA is for seeing what the data can tell us beyond the formal modeling or hypothesis testing task.
# 

# ## **2.1** EDA

# ### **2.1.1** Understanding Data
# 
# First of all, we need to understand the data, through descriptive statistics, datatypes, or just a quick tabular visualization.

# In[ ]:


# Display the datatypes in `df_foodfacts`
df_foodfacts_cleaned.dtypes


# In[ ]:


# Display the descriptive statistics of `df_reservations`
df_foodfacts_cleaned.describe()


# In[ ]:


df_foodfacts_cleaned.info()


# In[ ]:


print(df_foodfacts_cleaned.columns.tolist())


# ### **2.1.2** Data Visualization
# 
# As a second step, we will use data visualization approach to detect some trends, characteristics, fun facts about open food data. Through some key questions and visualization to demonstrate the answer to those questions, we have a better undersatanding of data distribution and key attributes accordingly.  

# 
# #### (a) Which country has the most number of products in this dataset?

# In[ ]:


# Assuming df_foodfacts is your main dataframe and 'countries_en' is the column with country data
countries_count = df_foodfacts_cleaned['countries_en'].value_counts().head(10).rename_axis('country').reset_index(name='count')

# Display the DataFrame with a background gradient
result = countries_count.style.background_gradient(cmap='Greens')
result


# **Conclusion**
# Through the table above, we find that developed countries tend to have larger number of food entries. United States has the largest number of food entries, it implies either the country has more food in reality or volunteers / developers in the United States tend to upload food entries more often.

# #### (b) Top 10 Most Common Additives

# In[ ]:


additives_series = df_foodfacts_cleaned['additives_en'].value_counts()

# Plotting the bar chart
plt.figure(figsize=(6, 4))
sns.barplot(y=additives_series.index[:10], x=additives_series.values[:10], palette='viridis')
plt.title('Top 10 Most Common Additives')
plt.xlabel('Frequency')
plt.ylabel('Additives')
plt.show()


# **Conclusion**
# 
# Through the bar plot above, we find out the top 10 most common used additives globally. Citric Acid is the most common additive. Based on common sense, we know that citric acid is widely used as a flavor enhancer in foods and beverages. The second common additive is lecthin, which is also used as an emulsifier in food products, helping stabilize mixtures of oil and water.

# #### (c) Top 10 Main Categories

# In[ ]:


main_category_series = df_foodfacts_cleaned['main_category_en'].value_counts()

# Plotting the bar chart
plt.figure(figsize=(6, 4))
sns.barplot(y=main_category_series.index[:10], x=main_category_series.values[:10], palette='viridis')
plt.title('Top 10 Most Common Categories')
plt.xlabel('Frequency')
plt.ylabel('Categories')
plt.show()


# #### (d) Average Nutrition Score by Country

# In order to present maps of nutritional quality in different countries and regions and to identify correlations between geographic location and nutritional quality of food products

# In[ ]:


pip install contextily


# In[ ]:


pip install adjustText


# In[ ]:


# remove the rows without 'nutrition-score-fr_100g'
df_foodfacts_nutri = df_foodfacts_cleaned.dropna(axis=0, subset=['nutrition-score-fr_100g'])

# Group by 'countries_en' and calculate count and mean for 'nutrition-score-fr_100g'
country_aggregates = df_foodfacts_nutri.groupby('countries_en').agg(
    count=('nutrition-score-fr_100g', 'count'),
    average_nutrition_score=('nutrition-score-fr_100g', 'mean')
).reset_index()

# Rename columns to better represent the data
country_aggregates.rename(columns={'countries_en': 'country'}, inplace=True)

# sort by average nutrition score
country_aggregates = country_aggregates.sort_values(by='average_nutrition_score', ascending=True)

# remove the countries with less than 30 rows
country_aggregates = country_aggregates[country_aggregates['count'] >= 30]

# Display the resulting DataFrame
print(country_aggregates)


# In[ ]:


get_ipython().system('pip install geopandas')


# In[ ]:


import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx

# Load the world map
world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))

# Prepare country_nutrition DataFrame
country_nutrition = country_aggregates[['country', 'average_nutrition_score']]

# Apply the name mapping to standardize country names if needed
name_mapping = {
    'United States': 'United States of America',
}
country_nutrition['country'] = country_nutrition['country'].replace(name_mapping)

# Merge the country names with the world map data
merged = world.merge(country_nutrition, how='left', left_on='name', right_on='country')

# Adjust the aspect ratio of the plot
fig, ax = plt.subplots(1, 1, figsize=(15, 8))  # Adjust figsize for aspect ratio

# Plot the world map with country outlines
world.boundary.plot(ax=ax, linewidth=1, edgecolor='black')

# Plot the countries with average nutrition score data
merged.dropna().plot(column='average_nutrition_score', ax=ax, legend=True,
                     legend_kwds={'label': "Average Nutrition Score by Country",
                                  'orientation': "horizontal"}, cmap='OrRd',
                     edgecolor='black', linewidth=0.1)  # Thin black line for country borders

# Set alpha to 0.6 for the patches with data
for patch in [p for p in ax.patches if p.get_facecolor() != (1, 1, 1, 1)]:  # Check for non-white patches
    patch.set_alpha(0.6)

# Turn off axis for a cleaner map
ax.set_axis_off()

# Set title
ax.set_title('Average Nutrition Score by Country (>= 30 Records)', fontsize=12)

plt.show()


# **Conclusion**
# 
# We find Eastern Europe, including Romania, Russia, Hungray, Poland tend to have food that have higher nutrition scores. Western Europe, Nordic Europe and Asia, for example, Spain, Austria, Sweden, Norway and Hong Kong tend to have food that have lower nutrition scores.  

# #### (e) Average Nurition Score by Food Category

# In[ ]:


# Group by 'countries_en' and calculate count and mean for 'nutrition-score-fr_100g'
category_aggregates = df_foodfacts_nutri.groupby('main_category_en').agg(
    count=('nutrition-score-fr_100g', 'count'),
    average_nutrition_score=('nutrition-score-fr_100g', 'mean')
).reset_index()

# Rename columns to better represent the data
category_aggregates.rename(columns={'main_category_en': 'main_category'}, inplace=True)

# sort by average nutrition score
category_aggregates = category_aggregates.sort_values(by='average_nutrition_score', ascending=True)

# remove the categories with less than 50 rows
category_aggregates = category_aggregates[category_aggregates['count'] >= 50]

# Display the resulting DataFrame
print(category_aggregates)


# In[ ]:


plt.figure(figsize=(12, 6))

sns.barplot(data=category_aggregates.drop("count", axis=1), x='main_category', y='average_nutrition_score', palette='husl')

# Set title and axis labels
plt.title('Average Nutrition Score by Food Category')
plt.xlabel('Food Category')
plt.ylabel('Average Nutrition Score')
plt.xticks(rotation=90, weight='normal')

plt.show()


# **Conclusion**
# Through the bar plot above, sugary snacks, waffles, fish and meat dogs, terrines, spreads, and pie dough have the highest average nutrition scores, suggesting that these categories may contain less healthy options based on the scoring criteria. Farming products and baby foods, tabbouleh, and beverages have the lowest average nutrition scores, indicating they might be considered healthier options.

# #### (f) Average Nurition Score by Nutrition Grade

# In[ ]:


# Group by 'countries_en' and calculate count and mean for 'nutrition-score-fr_100g'
nutrition_grade_aggregates = df_foodfacts_nutri.groupby('nutrition_grade_fr').agg(
    count=('nutrition-score-fr_100g', 'count'),
    average_nutrition_score=('nutrition-score-fr_100g', 'mean')
).reset_index()

# Rename columns to better represent the data
nutrition_grade_aggregates.rename(columns={'nutrition_grade_fr': 'nutrition_grade'}, inplace=True)

# sort by average nutrition score
nutrition_grade_aggregates = nutrition_grade_aggregates.sort_values(by='average_nutrition_score', ascending=True)


# Display the resulting DataFrame
print(nutrition_grade_aggregates)


# In[ ]:


plt.figure(figsize=(6, 4))

sns.barplot(data=nutrition_grade_aggregates.drop("count", axis=1), x='nutrition_grade', y='average_nutrition_score', palette='husl')

# Set title and axis labels
plt.title('Average Nutrition Score by Nutrition Grade')
plt.xlabel('Nutrition Grade')
plt.ylabel('Average Nutrition Score')

plt.show()


# **Conclusion**
# 
# From the chart above we can see that the lower nutrition scores have a higher nutrition grade. In other words, Grade A tends to have lowest average nutrition scores, while Grade E tends to have highest average nutrition scores.

# #### (g) Average Nurition Score Comparision Between Organic vs Non-Organic Foods

# Firstly, we need to judge whether a food product is organic or not. After roughly reviewing data, we find that 3 columns may contain information whether the food is organic - product_name, ingredients_text, additives.
# 
# We create a column "organic" - 1 if any of these three columns contain text "organic" otherwise 0.
# 
# We continue to use df_foodfacts_nutri (drop the rows without nutrition scores)

# In[ ]:


# Convert all relevant columns to string type
df_foodfacts_nutri['product_name'] = df_foodfacts_nutri['product_name'].astype(str)
df_foodfacts_nutri['ingredients_text'] = df_foodfacts_nutri['ingredients_text'].astype(str)
df_foodfacts_nutri['additives_en'] = df_foodfacts_nutri['additives_en'].astype(str)

# Check if the word 'organic' appears in any of the three columns
df_foodfacts_nutri['organic'] = df_foodfacts_nutri.apply(lambda x: 1 if 'organic' in x['product_name'].lower() or
                                                     'organic' in x['ingredients_text'].lower() or
                                                     'organic' in x['additives_en'].lower()
                                                else 0, axis=1)

# Group by 'countries_en' and calculate count and mean for 'nutrition-score-fr_100g'
organic_aggregates = df_foodfacts_nutri.groupby('organic').agg(
    count=('organic', 'count'),
    average_nutrition_score=('nutrition-score-fr_100g', 'mean')
).reset_index()

print(organic_aggregates)


# **Conclusion**
# 
# Based upon our analysis, the average nutrition score of organic food is 5.85, comparing to the average nutrition score of non-organic food is 9.40. It also demonstrates that organic food tends to be healthier food options with lower nutrition scores, and that non-organic food usually have higher nutrition scores.

# ##### Comparision of Nutrition Score for Organic vs Non-Organic Food in Top 10 Category

# In[ ]:


organic_aggregates_by_cate = df_foodfacts_nutri[df_foodfacts_nutri['main_category_en'].isin(list(main_category_series.index[:10]))].groupby(['organic', 'main_category_en']).agg(
    count=('organic', 'count'),
    average_nutrition_score=('nutrition-score-fr_100g', 'mean')
).reset_index()

print(organic_aggregates_by_cate)


# In[ ]:


plt.figure(figsize=(15, 6))

sns.barplot(data=organic_aggregates_by_cate.drop("count", axis=1), x='main_category_en', y='average_nutrition_score', hue='organic', palette='husl')

# Set title and axis labels
plt.title('Comparision of Nutrition Score for Organic vs Non-Organic Food in Top 10 Category')
plt.xlabel('Food Category')
plt.ylabel('Average Nutrition Score')

plt.show()


# ####(h) Distribution of nutrient scores

# Distributions of Nutrition Score

# In[ ]:


# Drop NA values from the new combined column for accurate visualization
data = df_foodfacts_cleaned['nutrition-score-fr_100g'].dropna()

# Setting up the figure
plt.figure(figsize=(12, 8))

# Create a subplot grid with 2 rows, 1 for each type of plot
plt.subplot(2, 1, 1)
# Plotting the histogram
sns.histplot(data, bins=30, kde=False, color='skyblue', edgecolor='black')
plt.title('Histogram of Nutrition Scores')
plt.xlabel('Nutrition Score')
plt.ylabel('Frequency')

# Adding a subplot for the boxplot
plt.subplot(2, 1, 2)
# Plotting the boxplot
sns.boxplot(x=data, color='lightgreen')
plt.title('Boxplot of Nutrition Scores')
plt.xlabel('Nutrition Score')

# Adding space between the subplots for clarity
plt.tight_layout(pad=3.0)

plt.show()


# **Conclusion**
# 
# Nutrition scores range from below 0 to above 40, but most scores fall between 0 and 30. The highest bars in the histogram suggest the most common range of nutrition scores is around 10 to 15.

# ## **2.2** Correlation of Feature Variables

# #### (a) Isolating Numerics from Categorical Features
# 
# Before anything else, it may help to create groups of the numeric and categorical variables.
# 
# We want to split the `df_foodfacts_nutri` dataframe into 2 dataframes:
# 
# 1. `numerics_df`: This dataframe contains all numerical columns from `df_foodfacts_nutri`
# 
# 2. `categorical_df`: This dataframe contains all categorical columns from `df_foodfacts_nutri`
#   - i.e. the columns of non-numeric type or contain boolean values

# In[ ]:


# Visualize number of unique values and datatype in each column (call .nunique())
df_foodfacts_nutri.nunique()


# In[ ]:


# show foodfacts data types
df_foodfacts_nutri.dtypes


# We need to reduce the number of numeric columns. We may remove the columns that have less than 10,000 rows with valid data.

# In[ ]:


# Calculate the number of non-null values in each column
nonnull_counts = df_foodfacts_nutri.notnull().sum()

# Identify columns where the number of non-nulls is less than or equal to 10,000
columns_to_drop = nonnull_counts[nonnull_counts <= 10000].index

# Drop these columns from the DataFrame
df_foodfacts_clean = df_foodfacts_nutri.drop(columns=columns_to_drop)

print(df_foodfacts_clean.shape)


# In[ ]:


# select the colums where the datatype is float64
numerics_df = df_foodfacts_clean.select_dtypes(include=['float64'])
numerics_df.head()

numerics_columns = list(numerics_df.columns)

# Print the columns of the new DataFrame to verify the types
print("Numerical columns:", numerics_df.columns)


# Next, we select categorical columns and exclude irrelevant columns.

# In[ ]:


# Select columns that are not float64
categorical_df_orig = df_foodfacts_clean.select_dtypes(exclude=['float64'])

# Print the columns of the new DataFrame to verify the types
print("Categorical columns:", categorical_df_orig.columns)


# Based upon reviewing catogrical columns, we only keep 'packaging_tags', 'brands_tags', 'categories_en',
#        'countries_en', 'serving_size', 'additives_en',
#        'nutrition_grade_fr', 'pnns_groups_1', 'pnns_groups_2', 'states_en',
#        'main_category_en', 'organic'.

# In[ ]:


# Select columns that are not of numeric types (removed description columns like product_name and ingredients_text)
categorical_df = categorical_df_orig[['packaging_tags', 'brands_tags', 'categories_en',
       'countries_en', 'serving_size', 'additives_en',
       'nutrition_grade_fr', 'pnns_groups_1', 'pnns_groups_2', 'states_en',
       'main_category_en', 'organic']]

# Print the columns of the new DataFrame to verify the types
categorical_df.nunique()


# According to the number of unique values and our basic knowledge on food, for future analysis on nutrition score and categories, we will include the following categorical columns:
# - 'countries_en', 'nutrition_grade_fr', 'pnns_groups_1' 'pnns_groups_2', 'organic'

# In[ ]:


categorical_df_final = categorical_df[['countries_en', 'nutrition_grade_fr', 'pnns_groups_1', 'pnns_groups_2','organic']]
categorical_columns = ['countries_en', 'nutrition_grade_fr', 'pnns_groups_1', 'pnns_groups_2','organic']


# #### (b) **Correlation Heatmap**
# 
# Create a correlation matrix using `numerics_df` and call it `corr_mat`. Using the correlation matrix, generate a correlation heatmap for these numeric features. We will be using Seaborn library to create this heatmap.

# In[ ]:


# create a correlation matrix
corr_mat = numerics_df.corr()
# Set figure size
plt.figure(figsize=(20, 20))
# create heatmap
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="RdBu", center=0, vmin=-1, vmax=1)

plt.title('Correlation Heatmap of Numeric Features')
plt.show()


# The Correlation Heatmap illustrates that the following columns are highly correlated:
# - 'salt_100g', 'sodium_100g'
# 
# Therefore, we could remove 'salt_100g'.

# In[ ]:


numerics_df_final = numerics_df.drop(columns = [ 'salt_100g'])

# Print the columns of the new DataFrame
print("Numerical columns:", numerics_df_final.columns)
numerics_columns = list(numerics_df_final.columns)


# ## **2.3** Feature Engineering

# Feature engineering is the process of applying domain knowledge to extract features from raw data via data mining techniques. These features can be used to improve the performance of machine learning algorithms. For nutrition score predition, need to select the right features from EDA & drop unrelated ones.

# ### **2.3.1** One Hot Encoding

# One hot encoding is a process by which categorical variables are converted into a form that could be provided to ML algorithms to do a better job in prediction. With one-hot encoding, we convert each categorical value into a new categorical column and assign a binary value of 1 or 0 to those columns.

# In[ ]:


# combining numeric and categorical columns
df_foodfacts_final = df_foodfacts_clean[numerics_columns + categorical_columns]
# df_foodfacts_final = df_foodfacts_clean[categorical_columns+['nutrition-score-fr_100g']]
print(df_foodfacts_clean.shape)


# In[ ]:


# TO-DO: create dataframe 'encoded_df_foodfacts' that contains the appropriate one hot encoded columns
encoded_df_foodfacts = pd.get_dummies(df_foodfacts_final[['countries_en', 'nutrition_grade_fr', 'pnns_groups_1', 'pnns_groups_2', 'organic']])
df_foodfacts_rest = df_foodfacts_final.drop(columns = ['countries_en', 'nutrition_grade_fr', 'pnns_groups_1', 'pnns_groups_2','organic'])
encoded_df_foodfacts = pd.concat([df_foodfacts_rest,encoded_df_foodfacts], axis = 1)


# In[ ]:


# CHECK: display the first 5 rows of 'encoded_df_foodfacts'
encoded_df_foodfacts.head()


# In[ ]:


# Downstampling to 60000 to optimize compute usage
encoded_df_foodfacts = encoded_df_foodfacts.sample(n=60000)


# # **Part III:** Regression Models

# ## **3.0.0** Preprocessing: Create Features and Label and Split Data into Train and Test
# The features will be all the variables in the dataset **except** `"nutrition-score-fr_100g"`, which will act as the label for our model. First, store these two as `features` (pd.DataFrame) and `target` (pd.Series), respectively.
# 

# In[ ]:


features = encoded_df_foodfacts.drop(columns = 'nutrition-score-fr_100g')
target = encoded_df_foodfacts['nutrition-score-fr_100g']

seed = 42
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state = seed)


# In[ ]:


# # Value Inmputation using KNN
# from sklearn.impute import KNNImputer
# imp = KNNImputer(n_neighbors=2)
# imp.fit(X_train)
# X_train_imp = imp.transform(X_train)
# X_test_imp = imp.transform(X_test)


# In[ ]:


# Value Inmputation
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
imp.fit(X_train)
X_train_imp = imp.transform(X_train)
X_test_imp = imp.transform(X_test)


# ## **3.0.1** PCA for regression
# 

# As a first step, we instantiate the `PCA` class from scikit-learn and fit it on the training set. The purpose of using PCA is to remove multicollinearity and prevent overfitting.

# In[ ]:


# Import necessary libraries
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


# In[ ]:


# Intermediate step to address fac that PCA is not scale-invariant

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scl = scaler.fit_transform(X_train_imp)
X_test_scl = scaler.transform(X_test_imp)

# Instantiate PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scl)


# #### **3.0.1.1** Cumulative Explained Variance Ratios

# 
# 
# Create an array of explained variance ratios and store it into a variable called `explained_variance_ratios`. Also, calculate the _cumulative_ explained variance ratios and store that into another variable called `cum_evr`.

# In[ ]:


# Save the explained variance ratios into variable called "explained_variance_ratios"

explained_variance_ratios = pca.explained_variance_ratio_
print(explained_variance_ratios)

# Save the CUMULATIVE explained variance ratios into variable called "cum_evr"
cum_evr = np.cumsum(explained_variance_ratios)
print(cum_evr)


# Now plot the _cumulative_ `explained_variance_ratio` against the number of components to decide the number of components we should keep. Also add a horizontal line that represents the 80% of the variance as a threshold.

# In[ ]:


# Assuming 'cum_evr' is a predefined list with cumulative explained variance ratios
components = list(range(1, len(cum_evr) + 1))  # Components from 1 to the number of items in cum_evr
plt.figure(figsize=(8, 6))
plt.plot(components, cum_evr, marker='o', linestyle='-', color='b', linewidth=1)  # Set linewidth to 1 for a thinner line

# Add a horizontal line at y=0.8 for the 80% threshold
plt.axhline(y=0.8, color='r', linestyle='--', label='80% explained variance')

# Add labels, title, and grid
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Optimal Number of PCA Components based on Explained Variance')
plt.grid(True)
plt.legend()

# Display the optimized plot
plt.show()


# #### **3.0.1.2** Final PCA

# We then use the results above to help decide the number of components to keep, choose a number (`n`) that explains **at least 80% of total variance** in the dataset. Then re-fit and transform PCA on the training set using the number of components we decided.

# In[ ]:


# Get transformed set of principal components on x_test

# 1. Refit and transform on training with parameter n
n = sum(cum_evr < 0.80) + 1
pca_optimal = PCA(n_components = n)
X_train_pca = pca_optimal.fit_transform(X_train_scl)

# 2. Transform on Testing Set and store it as `X_test_pca`
X_test_pca = pca_optimal.transform(X_test_scl)
print(n)


# ## 3.1 Linear Regression

# First of all, we use a Linear Regression model to the PCA-transformed training data, to predict outcomes for the PCA-transformed test data, and calculates the mean squared error (MSE) and R-squared (R2) metrics to evaluate the model's performance in terms of error magnitude and the proportion of variance explained by the model, respectively.

# In[ ]:


from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Instantiate the Linear Regression model
linear_reg = LinearRegression()

# Fit the model on the PCA-transformed training data
linear_reg.fit(X_train_pca, y_train)

# Predict on the PCA-transformed test data
y_pred_linear = linear_reg.predict(X_test_pca)

# Manually enforcing predictions within range [-15,40] by setting any values below -15 to -15 and any values above 40 to 40
for i in range(0,len(y_pred_linear)):
  if y_pred_linear[i]<-15:
    y_pred_linear[i]=-15
  if y_pred_linear[i]>40:
    y_pred_linear[i]=40
  i+=1


# Calculate metrics
mse_linear = mean_squared_error(y_test, y_pred_linear)
r2_linear = r2_score(y_test, y_pred_linear)


print("Linear Regression MSE:", mse_linear)
print("Linear Regression R2 Score:", r2_linear)


# ## 3.2 Ridge Regression

# As a second step, we wet up and train a Ridge Regression (L2) model with regularization to prevent overfitting, then evaluate its performance using mean squared error and R-squared metrics after making predictions on the PCA-transformed test data. We manually enforce predictions within range [-15,40] by setting any values below -15 to -15 and any values above 40 to 40

# In[ ]:


from sklearn.linear_model import Ridge

# Instantiate the Ridge Regression model
ridge_reg = Ridge(alpha=10)

# Fit the model
ridge_reg.fit(X_train_pca, y_train)

# Predict
y_pred_ridge = ridge_reg.predict(X_test_pca)

# Manually enforcing predictions within range [-15,40] by setting any values below -15 to -15 and any values above 40 to 40
for i in range(0,len(y_pred_ridge)):
  if y_pred_ridge[i]<-15:
    y_pred_ridge[i]=-15
  if y_pred_ridge[i]>40:
    y_pred_ridge[i]=40
  i+=1

# Calculate metrics
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print("Ridge Regression MSE:", mse_ridge)
print("Ridge Regression R2 Score:", r2_ridge)


# Linear Regression and Ridge Regression turn out to have similar MSE and R2 Score, which indicates that Ridge Regression doesn't improve the linear regression. Also, given that we have already done StandardScaler and PCA, it could avoid the situation that some features may have very significant parameters.

# ## 3.3 Random Forest Regression

# Thirdly, we initialize and train a Random Forest Regressor model, then uses it to predict outcomes on the PCA-transformed test dataset, subsequently evaluating its accuracy and explanatory power through the mean squared error and R-squared metrics.

# In[ ]:


from sklearn.ensemble import RandomForestRegressor

# Instantiate the Random Forest model
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit the model
rf_reg.fit(X_train_pca, y_train)

# Predict
y_pred_rf = rf_reg.predict(X_test_pca)

# Manually enforcing predictions within range [-15,40] by setting any values below -15 to -15 and any values above 40 to 40
for i in range(0,len(y_pred_rf)):
  if y_pred_rf[i]<-15:
    y_pred_rf[i]=-15
  if y_pred_rf[i]>40:
    y_pred_rf[i]=40
  i+=1

# Calculate metrics
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print("Random Forest Regression MSE:", mse_rf)
print("Random Forest Regression R2 Score:", r2_rf)


# It shows that Random Forest Regression results in a better model than Linear Regression and Ridge Regression. R2 Score is significantly higher than those with Linear Regression and Ridge Regression.

# # **Part IV:** Classification Models - Organic or Not

# ### 4.0.0 Preprocessing: Create Featuers and Label, Split Data into Train and Test, and Balance Data
# The features will be all the variables in the dataset **except** `"organic"`, which will act as the label for our model. First, store these two as `features` (pd.DataFrame) and `target` (pd.Series), respectively.
# 
# As we have seen in (g) of 2.1.2, non-organic class has much more data than organic class. Therefore, we will perform over-sampling on the minority class (organic) to balance data

# In[ ]:


pip install -U imbalanced-learn


# In[ ]:


# Over sampling
from imblearn.over_sampling import RandomOverSampler, SMOTE
#from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter


# In[ ]:


encoded_df_foodfacts_2 = encoded_df_foodfacts

features = encoded_df_foodfacts.drop(columns = 'organic')
target = encoded_df_foodfacts['organic']

seed = 42
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state = seed)


# In[ ]:


# Check the number of records before over sampling
print(sorted(Counter(y_train).items()))


# In[ ]:


# Randomly over sample the minority class in training dataset
ros = RandomOverSampler(random_state=101)
X_train_ros, y_train_ros= ros.fit_resample(X_train, y_train)

# Check the number of records after over sampling
print(sorted(Counter(y_train_ros).items()))


# In[ ]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
imp.fit(X_train)
X_train_imp = imp.transform(X_train_ros)
X_test_imp = imp.transform(X_test)


# ### 4.0.1 PCA for Classification
# 
# As a first step, we instantiate the `PCA` class from scikit-learn and fit it on the training set. The purpose of using PCA is to remove multicollinearity and prevent overfitting.

# In[ ]:


# Intermediate step to address fac that PCA is not scale-invariant

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scl = scaler.fit_transform(X_train_imp)
X_test_scl = scaler.transform(X_test_imp)

# Instantiate PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scl)


# #### **4.0.1.1** Cumulative Explained Variance Ratios

# 
# 
# Create an array of explained variance ratios and store it into a variable called `explained_variance_ratios`. Also, calculate the _cumulative_ explained variance ratios and store that into another variable called `cum_evr`.

# In[ ]:


# Save the explained variance ratios into variable called "explained_variance_ratios"
explained_variance_ratios = pca.explained_variance_ratio_
print(explained_variance_ratios)

# Save the CUMULATIVE explained variance ratios into variable called "cum_evr"
cum_evr = np.cumsum(explained_variance_ratios)
print(cum_evr)


# Now plot the _cumulative_ `explained_variance_ratio` against the number of components to decide the number of components we should keep. Also add a horizontal line that represents the 80% of the variance as a threshold.

# In[ ]:


# Assuming 'cum_evr' is a predefined list with cumulative explained variance ratios
components = list(range(1, len(cum_evr) + 1))  # Components from 1 to the number of items in cum_evr
plt.figure(figsize=(8, 6))
plt.plot(components, cum_evr, marker='o', linestyle='-', color='b', linewidth=1)  # Set linewidth to 1 for a thinner line

# Add a horizontal line at y=0.8 for the 80% threshold
plt.axhline(y=0.8, color='r', linestyle='--', label='80% explained variance')

# Add labels, title, and grid
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Optimal Number of PCA Components based on Explained Variance')
plt.grid(True)
plt.legend()

# Display the optimized plot
plt.show()


# #### **4.0.1.2** Final PCA

# We then use the results above to help decide the number of components to keep, choose a number (`n`) that explains **at least 80% of total variance** in the dataset. Then re-fit and transform PCA on the training set using the number of components we decided.

# In[ ]:


# Get transformed set of principal components on x_test

# 1. Refit and transform on training with parameter n
n = sum(cum_evr < 0.80) + 1
pca_optimal = PCA(n_components = n)
X_train_pca = pca_optimal.fit_transform(X_train_scl)

# 2. Transform on Testing Set and store it as `X_test_pca`
X_test_pca = pca_optimal.transform(X_test_scl)
print(n)


# ### 4.1 Logistic Regression

# First of all, we train a Logistic Regression model on X_train and y_train. We also calculate the accuracy of the model on the test set to evaluate the model performance.

# In[ ]:


# Import required libraries
import sklearn
from sklearn.linear_model import LogisticRegression

# Initialize model with default parameters and fit it on the training set
clf = LogisticRegression()
clf.fit(X_train_pca,y_train_ros)

# Use the model to predict on the test set and save these predictions as `y_pred`
y_pred = clf.predict(X_test_pca)

# Find the accuracy and store the value in `log_acc`
log_acc = clf.score(X_test_pca,y_test)
print("Accuracy: %.1f%%"% (log_acc*100))


# ### 4.2 Random Forest Classifier

# Secondly, we fit a Random Forest Classifier on the X_train and y_train with the following hyperparameters:
# - balanced class_weight
# - 120 estimators
# - maximum depth of 30
# - random seed set to 42
# 
# We also calculate the accuracy of the model on the test set using the score method.

# In[ ]:


# Import required libraries
from sklearn.ensemble import RandomForestClassifier

# Initialize model with default parameters and fit it on the training set
clf = RandomForestClassifier(n_estimators = 120, max_depth=30, random_state=42, class_weight='balanced')
clf.fit(X_train_pca,y_train_ros)

# Use the model to predict on the test set and save these predictions as `y_pred`
y_pred = clf.predict(X_test_pca)

# Find the accuracy and store the value in `rf_acc`
rf_acc = clf.score(X_test_pca,y_test)
print("Accuracy: %.1f%%"% (rf_acc*100))

# TO-DO: Compute the confusion matrix and save it to `rf_confusion`
rf_confusion = sklearn.metrics.confusion_matrix(y_test, y_pred)


# Random Forest Classifier result in an accuracy score of 92.2%, as compared to 66.5% for Logistic Regression. Therefore, Random Forest Classifier has a better model performance.

# ##4.3 XGBoost

# ###4.3.1 XGBoost Model Training

# 
# We are training and evaluating an XGBoost classifier to predict product categories using preprocessed food data, and assessing the model's accuracy and effectiveness through various performance metrics.

# In[ ]:


get_ipython().system('pip install xgboost')
get_ipython().system('pip install imbalanced-learn')


# In[ ]:


# Import necessary libraries
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler

# Train the XGBoost classifier
model = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
model.fit(X_train_scl, y_train_ros)

# Make predictions
y_pred = model.predict(X_test_scl)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", report)


# Precision:
# 1. 0.98 for non-organic products (labeled 0): very high, indicating that the model almost never incorrectly labels organic products as non-organic when predicting non-organic products.
# 2. For organic products (labeled 1) it is 0.25: relatively low, indicating that the model has only about a 25% chance of being correct when predicting a product as organic.
# 
# Recall:
# 1. 0.86 for non-organic products: indicates that the model finds 86% of non-organic products.
# 2. 0.70 for organic: Better, meaning the model finds 70% of organic products.
# 
# F1-Score:
# 1. For non-organic products is 0.91: Higher, indicating a better balance of accuracy and recall.
# 2. For organic products it is 0.36: low, indicating that although the recall is fair, the accuracy is low, affecting the F1-score.

# ###Conclusion
# The overall accuracy is high, mainly due to the very accurate predictions for non-organic products, which make up the majority of the dataset.

# # **Part V:** Classification Models - PNNS Group

# ### 5.0.0 Preprocessing: Create Featuers and Label, Split Data into Train and Test, and Balance Data
# The features will be all the variables in the dataset **except** `"pnns_groups_1"`, which will act as the label for our model. First, store these two as `features` (pd.DataFrame) and `target` (pd.Series), respectively.

# In[ ]:


pip install -U imbalanced-learn


# In[ ]:


# Over sampling
from imblearn.over_sampling import RandomOverSampler, SMOTE
#from imblearn.under_sampling import RandomUnderSampler, NearMiss
from collections import Counter


# Rebuild data since we have one-hot coded 'pnns_groups_1'. Instead, we restart with df_foodfacts_final and keep 'pnns_groups_1'.

# In[ ]:


encoded_df_foodfacts_2 = pd.get_dummies(df_foodfacts_final[['countries_en', 'nutrition_grade_fr', 'pnns_groups_2', 'organic']])
df_foodfacts_rest_2 = df_foodfacts_final.drop(columns = ['countries_en', 'nutrition_grade_fr',  'pnns_groups_2','organic'])
encoded_df_foodfacts_2 = pd.concat([df_foodfacts_rest_2,encoded_df_foodfacts_2], axis = 1)


# First of all, we observe the data first.

# In[ ]:


print(encoded_df_foodfacts_2['pnns_groups_1'].value_counts().sort_index())


# We should remove unknown and NaN rows. It's meaningless to predict the category is unknown.

# In[ ]:


encoded_df_foodfacts_2 = encoded_df_foodfacts_2.dropna(subset=['pnns_groups_1'])
encoded_df_foodfacts_2 = encoded_df_foodfacts_2[encoded_df_foodfacts_2['pnns_groups_1'] != 'unknown']


# Then we split data to features and target and split further into training and test set for further operations.

# In[ ]:


features = encoded_df_foodfacts_2.drop(columns = 'pnns_groups_1')
target = encoded_df_foodfacts_2['pnns_groups_1']

seed = 42
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state = seed)


# In[ ]:


# Randomly over sample the minority class in training dataset
ros = RandomOverSampler(random_state=101)
X_train_ros, y_train_ros= ros.fit_resample(X_train, y_train)

# Check the number of records after over sampling
print((Counter(y_train_ros).items()))


# In[ ]:


from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
imp.fit(X_train)
X_train_imp = imp.transform(X_train_ros)
X_test_imp = imp.transform(X_test)


# ### 5.0.1 PCA for Classification

# In[ ]:


# Intermediate step to address fact that PCA is not scale-invariant

# Scale the data using StandardScaler
scaler = StandardScaler()
X_train_scl = scaler.fit_transform(X_train_imp)
X_test_scl = scaler.transform(X_test_imp)

# Instantiate PCA
pca = PCA()
X_train_pca = pca.fit_transform(X_train_scl)


# #### **5.0.1.1** Cumulative Explained Variance Ratios

# 
# 
# Create an array of explained variance ratios and store it into a variable called `explained_variance_ratios`. Also, calculate the _cumulative_ explained variance ratios and store that into another variable called `cum_evr`.

# In[ ]:


# Save the explained variance ratios into variable called "explained_variance_ratios"

explained_variance_ratios = pca.explained_variance_ratio_
print(explained_variance_ratios)

# Save the CUMULATIVE explained variance ratios into variable called "cum_evr"
cum_evr = np.cumsum(explained_variance_ratios)
print(cum_evr)


# Now plot the _cumulative_ `explained_variance_ratio` against the number of components to decide the number of components we should keep. Also add a horizontal line that represents the 80% of the variance as a threshold.

# In[ ]:


# Assuming 'cum_evr' is a predefined list with cumulative explained variance ratios
components = list(range(1, len(cum_evr) + 1))  # Components from 1 to the number of items in cum_evr
plt.figure(figsize=(8, 6))
plt.plot(components, cum_evr, marker='o', linestyle='-', color='b', linewidth=1)  # Set linewidth to 1 for a thinner line

# Add a horizontal line at y=0.8 for the 80% threshold
plt.axhline(y=0.8, color='r', linestyle='--', label='80% explained variance')

# Add labels, title, and grid
plt.xlabel('Number of PCA Components')
plt.ylabel('Cumulative Explained Variance Ratio')
plt.title('Optimal Number of PCA Components based on Explained Variance')
plt.grid(True)
plt.legend()

# Display the optimized plot
plt.show()


# #### **5.0.1.2** Final PCA

# We then use the results above to help decide the number of components to keep, choose a number (`n`) that explains **at least 80% of total variance** in the dataset. Then re-fit and transform PCA on the training set using the number of components we decided.

# In[ ]:


# Get transformed set of principal components on x_test

# 1. Refit and transform on training with parameter n
n = sum(cum_evr < 0.80) + 1
pca_optimal = PCA(n_components = n)
X_train_pca = pca_optimal.fit_transform(X_train_scl)

# 2. Transform on Testing Set and store it as `X_test_pca`
X_test_pca = pca_optimal.transform(X_test_scl)
print(n)


# ### 5.1 Logistic Regression

# First of all, we train a Logistic Regression model on X_train and y_train. We also calculate the accuracy of the model on the test set to evaluate the model performance.

# In[ ]:


# Import required libraries
import sklearn
from sklearn.linear_model import LogisticRegression

# Initialize model with default parameters and fit it on the training set
clf = LogisticRegression()
clf.fit(X_train_pca,y_train_ros)

# Use the model to predict on the test set and save these predictions as `y_pred`
y_pred = clf.predict(X_test_pca)

# Find the accuracy and store the value in `log_acc`
log_acc = clf.score(X_test_pca,y_test)
print("Accuracy: %.1f%%"% (log_acc*100))


# ### 5.2 Random Forest Classifier

# Secondly, we fit a Random Forest Classifier on the X_train and y_train with the following hyperparameters:
# - balanced class_weight
# - 120 estimators
# - maximum depth of 30
# - random seed set to 42
# 
# We also calculate the accuracy of the model on the test set using the score method.

# In[ ]:


# Import required libraries
from sklearn.ensemble import RandomForestClassifier

# Initialize model with default parameters and fit it on the training set
clf = RandomForestClassifier(n_estimators = 120, max_depth=30, random_state=42, class_weight='balanced')
clf.fit(X_train_pca,y_train_ros)

# Use the model to predict on the test set and save these predictions as `y_pred`
y_pred = clf.predict(X_test_pca)

# Find the accuracy and store the value in `rf_acc`
rf_acc = clf.score(X_test_pca,y_test)
print("Accuracy: %.1f%%"% (rf_acc*100))

# TO-DO: Compute the confusion matrix and save it to `rf_confusion`
rf_confusion = sklearn.metrics.confusion_matrix(y_test, y_pred)


# From the above analysis, we find that Random Forest Classifier has an accuracy score of 99.0%, which is slightly lower than Logistic Regression.

# # **Part VI:** Neural Network Modeling

# ###6.1 Neural Network for PNNS Group Prediction

# ###6.1.1 Build the Neural Network Model

# Configure a neural network model using TensorFlow and Keras, integrating layers with dropout for regularization, and compiles the model with Adam optimizer and binary crossentropy loss to prepare it for binary classification training and evaluation.

# In[ ]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Ensure data types are correct
X_train = X_train_imp.astype('float32')

# Convert labels to one-hot encoding
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train_ros)
y_train = to_categorical(y_train_encoded)

num_classes = y_train.shape[1]

# Build the neural network model
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_imp.shape[1],)),
    Dropout(0.3),
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Add early stopping mechanism to avoid overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)


# ###6.2.1 Train the Model

# In[ ]:


history = model.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_split=0.2,
    callbacks=[early_stopping]
)


# ###6.2.2 Evaluate the Model

# In[ ]:


# prepare test data
X_test = X_test_imp.astype('float32')
# y_test_encoded = label_encoder.transform(y_test)
# y_test = to_categorical(y_test_encoded)

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


# ###6.2.3 Visualization

# In[ ]:


import matplotlib.pyplot as plt

# Plotting training and validation accuracy
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

# Show the plot
plt.show()


# The key takeaway from this chart is that while the model is performing exceptionally well on the training data, it is not performing nearly as well on the validation data, indicating potential overfitting and a need for model adjustments to improve its generalization.
