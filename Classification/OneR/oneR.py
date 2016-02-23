'''
02/23/2016
oneR Classification
Demonstration of a oneR classification process.
Skeleton code copied from Yuya's implementation of zeroR
To anyone who plans to use this code, make sure you understand what exactly the code does before applying it.
If you plan to use it on another data set, some values will have to be adjusted.
'''
import os
import numpy as np
import collections
from StringIO import StringIO

# Kidney Disease Sample Dataset
file_dir = '../../datasets/kidney_disease/Kidney-Disease.csv'

data = np.genfromtxt(file_dir, delimiter=',', dtype="|S10")

#Function that finds a rule for a particular attribute.  It goes through the test data, finds the most common target class
#for each value of that attribute and associates the two in the rule{} dictionary which it returns. 
def FindRule(atts_with_targetvals, attribute_vals):
	attribute_vals_set = set(attribute_vals) #Construct a set of the attibute values (identical to attribute_vals but w/o duplicates)
	rule = {} #Rule will be the dictionary with a prediction for each attribute value

	#iterate over the set of attribute vals
	for val in attribute_vals_set:
		#Count how many times that attribute value got each target value
		class_counts = collections.Counter(value_pair[1] for value_pair in atts_with_targetvals if value_pair[0]== val)
		#The attribute value will correspond to its most common target value in the rule
		rule[val] = class_counts.most_common(1)[0][0]
	return rule

attribute_labels = data[0]
data = np.delete(data, (0), axis=0)   # Remove top row of attribute labels 

#Partition data into ten parts
slice_size = np.shape(data)[0] // 10

partition = np.array_split(data, slice_size) 

#Will become a list of the total number of correct predictions for each attribute
prediction_freq = [0] * (np.shape(data)[1] - 1)
#Will become a list of the number of values that had to be skipped in each attribute since there was no rule for them
skipped_freq = [0] * (np.shape(data)[1] - 1)

#10 Fold Cross-validation 
for i, cut in enumerate(partition):
	
	test = cut #test on one slice of the partition
	trainers = np.delete(partition, i, axis = 0) #train on the other 9 slices

	#Stack the rows of the test slices on top of one another into a single numpy array and store it in train
	train = partition[0]
	for j in range(1, np.shape(trainers)[0]):
		train = np.vstack((train, trainers[j]))
			
	#Pull out the target class labels
	train_labels = train[:, 24]
	test_labels  = test[:, 24]

	rules = [] #list that will store the rule dictionary for each attribute

	#GENERATE RULES
	#For every attribute in the data set (attributes correspond to columns)
	for attribute in range(0, data.shape[1] - 1):
		attribute_values = train[:, attribute] #Store the column corresponding to the current attribute into the list attribute_values
		attribute_and_target = zip(attribute_values, train_labels) #Make dict where each attribute_value corresponds to a target class label
		rules.append(FindRule(attribute_and_target, attribute_values)) #Find the rule for this attribute and store it in rules

	prediction_counts = [] #List to store the number of correct predictions for each attribute
	num_skipped = [] #List to keep track of the number of values that did

	#COMPUTE ACCURACY RATES 
	#For each attribute
	for attribute in range(0, data.shape[1] - 1):
		correct_predictions = 0
		values_skipped = 0
		attribute_values = test[:, attribute] #Store the column corresponding to the current attribute into the list attribute_values
		attribute_and_target = zip(attribute_values, test_labels) #Make dict where each attribute_value corresponds to a target class label
		for pair in attribute_and_target:
			if pair[0] not in rules[attribute].keys(): #If the value is not in the rule dictionary, skip it and add one to values_skipped
				values_skipped += 1
				continue
			if rules[attribute][pair[0]] == pair[1]:
				correct_predictions += 1
		prediction_counts.append(correct_predictions)
		num_skipped.append(values_skipped)
	for j in range(0,len(prediction_counts)):
		prediction_freq[j] += prediction_counts[j]
	for j in range(0, len(num_skipped)):
		skipped_freq[j] += num_skipped[j]

#Prepare results to display
freq_by_attribute = zip(attribute_labels, prediction_freq)
skipped_by_attribute = zip(attribute_labels, skipped_freq) 
winner = prediction_freq.index(max(prediction_freq))
acc = float(freq_by_attribute[winner][1]) / np.shape(data)[0]

# Display Test Results
print '[oneR Test Results]'
print 'Number of correct predictions for each attribute: '  
print freq_by_attribute
print 'Number of unclassifiable persons for each attribute: '
print skipped_by_attribute
print 'Attribute with most predictive power: ', freq_by_attribute[winner]
print 'Accuracy: ', acc
