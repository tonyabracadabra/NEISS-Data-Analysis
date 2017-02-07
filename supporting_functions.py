#! /usr/bin/env python

import pandas as pd
import numpy as np
import operator
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import train_test_split
from tensorflow.python.framework import ops
from DeepFeatureSelection.dfs2 import DeepFeatureSelectionNew

bodyParts = pd.read_csv("./BodyParts.csv")
diagnosisCodes = pd.read_csv("./DiagnosisCodes.csv")
disposition = pd.read_csv("./Disposition.csv")
neiss2014 = pd.read_csv("./NEISS2014.csv")

# The column names of the data
column_names = neiss2014.columns

"""
	Read three dictionary files
"""
# Body part
bp_dict = {i[1]:i[0] for i in bodyParts.values}
# Diagnosis
diag_dict = {i[1]:i[0] for i in diagnosisCodes.values}
# Dispositions
disp_dict = {i[1]:i[0] for i in disposition.values}

dicts = {'body_part':bp_dict, 'diag':diag_dict, 'disposition':disp_dict}

# Get the clean data and select a class for feature selection
def getCleanData(labelY):
	cleanData = neiss2014.copy()
	months = [i.split('/')[0] for i in neiss2014['trmt_date'].values]
	cleanData['trmt_date'] = months
	cleanData['trmt_date'] = months
	for column in ['stratum','sex']:
	    mapping = {j:i for i,j in enumerate(np.unique(neiss2014[column]))}
	    cleanData[column] = [mapping[i] for i in neiss2014[column]]
	cleanData.drop(cleanData.columns[[0,7,8,10,16,17]], axis=1, inplace=True)

	# Get label
	mapping = {j:i for i,j in enumerate(np.unique(cleanData[labelY]))}
	y = np.array([mapping[i] for i in cleanData[labelY]])
	cleanData.drop(labelY, 1)
	mm = MinMaxScaler(feature_range=(0, 1))

	# Get scaled data
	scaledData = mm.fit_transform(cleanData.values)

	return scaledData, y

# Get the table
def getTable():
	return neiss2014.copy()

# Get all dictionaries
def getDicts():
	return dicts

def getColumnData(column, indexes=None):
	if indexes == None:
		return neiss2014[column].values
	else:
		return neiss2014[column].values[indexes]

def getHiLoFreqPerCol(column, HighestOrLowest, number):
	""" Get the higest or lowest frequent item in a given column

    Parameters
    ----------
    column: string
        The name of a column in the data

    HighestOrLowest: string
    	Can be choose from 'highest' to 'lowest'

    number: int
    	number of top items either in descending or ascending order

    """

	data = getColumnData(column).tolist()
	try:
		theDict = dicts[column]
	except KeyError:
		theDict = None

	# Get frequency counts
	freqCounts = {key:data.count(key) for key in theDict.keys()}
	sortedFreqCounts = sorted(freqCounts.items(), key=operator.itemgetter(1))

	if HighestOrLowest == 'highest':
		try:
			return [theDict[i[0]] for i in sortedFreqCounts[-number:]]
		except:
			return sortedFreqCounts[-number:]
	elif HighestOrLowest == 'lowest':
		try:
			return [theDict[i[0]] for i in sortedFreqCounts[:number]]
		except:
			return sortedFreqCounts[:number]
	else:
		raise ValueError('Must input highest or highest for this function!')

# This is a function that given the name of the column, return the frequency of
# each component in the column
def calcItemsFreq(data):
	""" Return frequency 

    Parameters
    ----------
    data: numpy array
    	Data wrapped in numpy array

    """

	uniqueItems = np.unique(data)
	data = data.tolist()
	freqDict = {}
	for item in uniqueItems:
		freqDict[item] = data.count(item)*1.0/len(data)
	return freqDict

# This is a function is used to retrieve the indexes of a given keywords 
# in the narrative field
def getKeywordIndexes(keyword, column='narrative'):
	""" Get all indexes in a given column that contains a given keyword

    Parameters
    ----------
    keyword: string
    	The keyword to be found in the data

    column: string
        The name of a column in the data, default with 'narrative'

    """

	data = getColumnData(column).tolist()
	keywordIndexes = []
	for index, words in enumerate(data):
		if re.search(keyword, words, re.IGNORECASE)!=None:
			keywordIndexes.append(index)
	return keywordIndexes


# This is a function is used to takes two columns, find the 
def getAwithHiLoB(columnA, columnB, itemInB, topN=1, freqOrCount='freq',HighestOrLowest='highest'):
	""" Get the higest or lowest rate/occurance time item in a given column, that meet the conditon
	that columnB == itemInB

    Parameters
    ----------
    columnA: string
        The name of a column in the data, columnA here

    columnB: string
        The name of a column in the data, columnB here

    itemInB: type depends on the data type in columnB
    	item to be find in columnB

    topN: int
    	topN items to return

    freqOrCount: string
    	Options with 'freq' and 'count', that takes either the frequency or the count
    	into the ranking

    HighestOrLowest: string
    	Can be choose from 'highest' to 'lowest'

    """

	try:
		theDict = dicts[columnA]
	except KeyError:
		theDict = None

	dataA, dataB = getColumnData(columnA), getColumnData(columnB)

	indexesB = set(np.where(dataB==itemInB)[0].tolist())

	indexesA = [(itemA,set(np.where(dataA==itemA)[0].tolist())) for itemA in np.unique(dataA)]

	if freqOrCount == 'freq':
		probDict = {i[0]:len(i[1].intersection(indexesB))*1.0/len(i[1]) for i in indexesA}
	elif freqOrCount == 'count':
		probDict = {i[0]:len(i[1].intersection(indexesB)) for i in indexesA}
	else:
		raise ValueError('Wrong parameter for freqOrCount')

	sortedDict = sorted(probDict.items(), key=operator.itemgetter(1))

	if HighestOrLowest == 'highest':
		try:
			return [theDict[i[0]] for i in sortedDict[-topN:]]
		except:
			return sortedDict[-topN:]
	elif HighestOrLowest == 'lowest':
		try:
			return [theDict[i[0]] for i in sortedDict[:topN]]
		except:
			return sortedDict[:topN]
	else:
		raise ValueError('Must input highest or highest for this function!')


def runDFS(inputX, inputY):

    weights = []

    for random_state in xrange(100):
        # Resplit the data
        X_train, X_test, y_train, y_test = train_test_split(inputX, inputY, test_size=0.2, random_state=random_state)
        
        # Change number of epochs to control the training time
        dfsMLP = DeepFeatureSelectionNew(X_train, X_test, y_train, y_test, n_input=1, hidden_dims=[5], learning_rate=0.01, \
                                             lambda1=0.001, lambda2=1, alpha1=0.001, alpha2=0, activation='tanh', \
                                             weight_init='uniform',epochs=80, optimizer='Adam', print_step=50)
        dfsMLP.train(batch_size=5000)
        print("Train finised for random state:" + str(random_state))
        weights.append(dfsMLP.selected_ws[0])

    return weights

    # The generated weights will be in the weights folder
    np.save("./dfs_weights.npy", weights)