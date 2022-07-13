# -*- coding: utf-8 -*-
"""
Created on Sun Sep 26 15:40:34 2021

@author: sahru
"""
import os
import tarfile
from six.moves import urllib

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
fetch_housing_data()


import pandas as pd

def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)



housing = load_housing_data()
housing.info()
housing.head()
housing.describe()
print(housing["ocean_proximity"].value_counts())

import matplotlib.pyplot as plt

#housing.hist(bins=50,figsize=(20,15))

import numpy as np
# def split_train_test(data, test_ratio):
#     shuffled_indices = np.random.permutation(len(data))
#     test_set_size = int(len(data) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return data.iloc[train_indices], data.iloc[test_indices]

# train_set, test_set = split_train_test(housing, 0.2)

# print(len(train_set))

#this works, but it is not perfect: if you run the program again, it will generate a
#different test set! Over time, you (or your Machine Learning algorithms) will get to
#see the whole dataset, which is what you want to avoid.





#A common
#solution is to use each instance’s identifier to decide whether or not it should go
#in the test set (assuming instances have a unique and immutable identifier). For
#example, you could compute a hash of each instance’s identifier and put that instance
#in the test set if the hash is lower or equal to 20% of the maximum hash value. This
##ensures that the test set will remain consistent across multiple runs, even if you
#refresh the dataset. The new test set will contain 20% of the new instances, but it will
#not contain any instance that was previously in the training set. Here is a possible
#implementation:

# from zlib import crc32
# def test_set_check(identifier, test_ratio):
#     return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32
# def split_train_test_by_id(data, test_ratio, id_column):
#     ids = data[id_column]
#     in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
#     return data.loc[~in_test_set], data.loc[in_test_set]

#the housing dataset does not have an identifier column. The simplest
#solution is to use the row index as the ID:

# housing_with_id = housing.reset_index() # adds an `index` column
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")


# If you use the row index as a unique identifier, you need to make sure that new data
# gets appended to the end of the dataset, and no row ever gets deleted. If this is not
# possible, then you can try to use the most stable features to build a unique identifier.
# For example, a district’s latitude and longitude are guaranteed to be stable for a few
# million years, so you could combine them into an ID like so:
    # housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
    # train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

# Scikit-Learn provides a few functions to split datasets into multiple subsets in various
# ways. The simplest function is train_test_split, which does pretty much the same
# thing as the function split_train_test defined earlier, with a couple of additional
# features. First there is a random_state parameter that allows you to set the random
# generator seed as explained previously, and second you can pass it multiple datasets
# with an identical number of rows, and it will split them on the same indices (this is
# very useful, for example, if you have a separate DataFrame for labels):
# from sklearn.model_selection import train_test_split
# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# stratifiedsampling: the population is divided into homogeneous subgroups called strata,
# and the right number of instances is sampled from each stratum to guarantee that the
# test set is representative of the overall population.

# The following code uses the
# pd.cut() function to create an income category attribute with 5 categories (labeled
# from 1 to 5): category 1 ranges from 0 to 1.5 (i.e., less than $15,000), category 2 from
# 1.5 to 3, and so on:

housing["income_cat"] = pd.cut(housing["median_income"],bins=[0., 1.5, 3.0, 4.5, 6., np.inf],labels=[1, 2, 3, 4, 5])
housing["income_cat"].hist()

# stratified sampling based on the income category. For this
# you can use Scikit-Learn’s StratifiedShuffleSplit class:

from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
print(strat_test_set["income_cat"].value_counts() / len(strat_test_set))

#Now you should remove the income_cat attribute so the data is back to its original
#state:
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)

#create a copy so you can play with it without harming the training set:    
housing = strat_train_set.copy()

#Looking for Correlations
# housing.plot(kind="scatter", x="longitude", y="latitude")
# This looks like California all right, but other than that it is hard to see any particular
# pattern. Setting the alpha option to 0.1 makes it much easier to visualize the places
# where there is a high density of data points


#housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)

#The radius of each circle represents
# the district’s population (option s), and the color represents the price (option c). We
# will use a predefined color map (option cmap) called jet, which ranges from blue
# (low values) to red (high prices):
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
s=housing["population"]/100, label="population", figsize=(10,7),
c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
)
plt.legend()



# compute the standard correlation
# coefficient (also called Pearson’s r) between every pair of attributes using the corr()
# method:

corr_matrix = housing.corr()
print(corr_matrix["median_house_value"].sort_values(ascending=False))

# The correlation coefficient only measures linear correlations (“if x
# goes up, then y generally goes up/down”). It may completely miss
# out on nonlinear relationships (e.g., “if x is close to zero then y generally
# goes up”).



# Another way to check for correlation between attributes is to use Pandas’
# scatter_matrix function, which plots every numerical attribute against every other
# numerical attribute. Since there are now 11 numerical attributes, you would get 112 =
# 121 plots, which would not fit on a page, so let’s just focus on a few promising
# attributes that seem most correlated with the median housing value
from pandas.plotting import scatter_matrix
# attributes = ["median_house_value", "median_income", "total_rooms",
# "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))

#The most promising attribute to predict the median house value is the median
#income, so let’s zoom in on their correlation scatterplot
# housing.plot(kind="scatter", x="median_income", y="median_house_value",
# alpha=0.1)

#Experimenting with Attribute Combinations

# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"]=housing["population"]/housing["households"]

# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))



# Prepare the Data for Machine Learning Algorithms
# let’s revert to a clean training set (by copying strat_train_set once again),
# and let’s separate the predictors and the labels since we don’t necessarily want to apply
# the same transformations to the predictors and the target values (note that drop()
# creates a copy of the data and does not affect strat_train_set):
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

# Data Cleaning
housing.dropna(subset=["total_bedrooms"]) # option 1
housing.drop("total_bedrooms", axis=1) # option 2
median = housing["total_bedrooms"].median() # option 3
housing["total_bedrooms"].fillna(median, inplace=True)

# Scikit-Learn provides a handy class to take care of missing values: SimpleImputer.

from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1) #median can only be computed on numerical attributes, we need to create a
# copy of the data without the text attribute ocean_proximity:

imputer.fit(housing_num)
#The imputer has simply computed the median of each attribute and stored the result
# in its statistics_ instance variable. Only the total_bedrooms attribute had missing
# values, but we cannot be sure that there won’t be any missing values in new data after
# the system goes live, so it is safer to apply the imputer to all the numerical attributes:

# print(imputer.statistics_)
# print(housing_num.median().values)

# Now you can use this “trained” imputer to transform the training set by replacing
# missing values by the learned medians:
    
X = imputer.transform(housing_num)

housing_tr = pd.DataFrame(X, columns=housing_num.columns)



#Handling Text and Categorical Attributes

housing_cat = housing[["ocean_proximity"]]
print(housing_cat.head(10))

# Most Machine Learning algorithms prefer to work with numbers anyway, so let’s convert
# these categories from text to numbers. For this, we can use Scikit-Learn’s Ordina
# lEncoder class

from sklearn.preprocessing import OrdinalEncoder
ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
print(housing_cat_encoded[:10])
print(ordinal_encoder.categories_)

# One issue with this representation is that ML algorithms will assume that two nearby
# values are more similar than two distant values. This may be fine in some cases (e.g.,
# for ordered categories such as “bad”, “average”, “good”, “excellent”), but it is obviously
# not the case for the ocean_proximity column (for example, categories 0 and 4 are
# clearly more similar than categories 0 and 1). To fix this issue, a common solution is
# to create one binary attribute per category: one attribute equal to 1 when the category
# is “<1H OCEAN” (and 0 otherwise), another attribute equal to 1 when the category is
# “INLAND” (and 0 otherwise), and so on. This is called one-hot encoding, because
# only one attribute will be equal to 1 (hot), while the others will be 0 (cold). The new
# attributes are sometimes called dummy attributes. Scikit-Learn provides a OneHotEn
# coder class to convert categorical values into one-hot vectors20:


# from sklearn.preprocessing import OneHotEncoder
# cat_encoder = OneHotEncoder()
# housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
# print(housing_cat_1hot)
# housing_cat_1hot.toarray()
# cat_encoder.categories_


#Custom Transformers





















