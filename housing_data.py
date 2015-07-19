import pandas as pd
import statsmodels.api as sm
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.cluster import KMeans

# we are going to try to predict the neighborhood based on the price/sqf of the apartment
# we are only going to examine coop buildings with elevators
# one major flaw is that there is no data on number of bedrooms


# retrieve data 
df = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')

# convert Sale Price to an integer
df['SALE_PRICE'] = df['SALE_PRICE'].map(lambda x: x.lstrip('$,,').rstrip('aAbBcC'))
df['SALE_PRICE'] = df['SALE_PRICE'].map(lambda x: x.replace(',', '').rstrip('aAbBcC'))
df['SALE_PRICE'] = df['SALE_PRICE'].map(lambda x: int(x))

# convert Land Square Feet to an integer
df['LAND_SQUARE_FEET'] = df['LAND_SQUARE_FEET'].map(lambda x: x.replace(',', '').rstrip('aAbBcC'))
df['LAND_SQUARE_FEET'] = df['LAND_SQUARE_FEET'].map(lambda x: int(x))

# format BUILDING CLASS CATEGORY by removing spaces and hyphen 
df['BUILDING_CLASS_CATEGORY'] = df['BUILDING_CLASS_CATEGORY'].map(lambda x: x.replace(' ', '').rstrip('aAbBcC'))
df['BUILDING_CLASS_CATEGORY'] = df['BUILDING_CLASS_CATEGORY'].map(lambda x: x.replace('-', '').rstrip('aAbBcC'))

# drop rows where SALE_PRICE and LAND_SQUARE_FEET are 0
df = df.loc[(df['SALE_PRICE'] != 0)]
df = df.loc[(df['LAND_SQUARE_FEET'] != 0)]

# create dummy variables to handle categorical variables of building type
building_type = pd.get_dummies(df['BUILDING_CLASS_CATEGORY'])
df = pd.merge(df, building_type, left_index=True, right_index=True)

# extract columns to use in the linear model
sqf = df['LAND_SQUARE_FEET']
price = df['SALE_PRICE']
coop_walkup = df['09COOPSWALKUPAPARTMENTS']
coop_elev = df['10COOPSELEVATORAPARTMENTS']

# reshape the data from the extracted columns
# use the sales price as the dependent variable
y = np.matrix(price).transpose()

# use the sqf, COOP walk-up and elevator categoricals as independent variables shaped as columns
x1 = np.matrix(sqf).transpose()
x2 = np.matrix(coop_walkup).transpose()
x3 = np.matrix(coop_elev).transpose()

# put the columns together to create an input matrix with one column per independent variable
x = np.column_stack([x1,x2,x3])

# create a linear model
X = sm.add_constant(x)
model = sm.OLS(y,X)
f = model.fit()

# output the results summary
f.summary()
# const	= 1.929e+07, x1	73.3860, x2	= -1.666e+07, x3 = 2.106e+07

# use k-NN to predict the COOP's neighborhood
# create training data set
dfTrain, dfTest = train_test_split(df, test_size=0.2)

# this is the subset of labels for the training set
cl = dfTrain[:,2]
# subset of labels for the test set, we're withholding these
true_labels  dfTest[:,2]

# apply k Nearest Neighbors to approximate which neighborhood an apartment came from
model = KNeighborsClassifier(n_neighbors=3)
model.fit(dfTrain[:,:2], dfTrain[:,2])

# we'll loop through and see what the misclassification rate is for different values of k
for k in range(1,20):
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(dfTrain[:,:2], dfTrain[:,2])
    # make predictions
    expected = dfTest[:,2]
    predicted = model.predict(dfTest[:,:2])
    # misclassification rate
    error_rate = (predicted != expected).mean()
    print('%d:, %.2f' % (k, error_rate))
	# we get a list ranging in misclassification rates of 11-15%
	# let's use a k of 4, this has the lowest misclassification rate and is low enough to prevent overfitting

# visualize results from k-NN
# see the default parameters below; you can initialize with none, any, or all of these.
model = KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances=True, verbose=0, random_state=None, copy_x=True, n_jobs=1)

# lets make a prediction for a 2,400sf COOP purchased for $2,900,000
predicted = model.predict([2400,2900000])
print predicted
# returns Washington Heights Lower