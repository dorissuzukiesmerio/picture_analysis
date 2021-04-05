#Categorizing imaegs by theme 
#Useful in search 

# Economic application:
# Apartment 
# Some type of maximizing 
# Categorizing thousands of images
# python -m install imageio
#IO - means input output

import imageio # to load dataset
import numpy
import os
import glob
import pandas
import matplotlib.pyplot as pyplot

from sklearn import preprocessing
from sklearn.mixture import GaussianMixture


def getrgb(filepath):
	imimage = imageio.imread('data/pic01.pjeg', pilmode='RGB') # this is the way to read colors
	#It will print the number of pixels
	# describing the color
	#2^8 = 255 binary . Maximun that can be contained in the 
	imimage_process = imimage_process/255
	imimage_process = imimage_process.sum(axis=0).sum(axis=0)/imimage_process.shape[1]
	#This is the number of columns and rows 
	# Will give an array of all the colors 

	#The results (numbers you get) is the sum of the color

	imimage_process = imimage_process/numpy.linalg.norm(imimage_process, ord=None) # WE are normalizing , in order to get percentages
	#the results will be percentage and will show:
	# red green blue - are the columns
	#average of all the 
	#intensity 
	# it is not compared to the red, green, blue
	# So:
	# light blue " will have a significant amount of blue, and sign of red and green too
	# dark blue : will have blue dominant, but not significant amount of red and blue
	print(imimage_process)
	return imimage_process # the function ends 

#Here, the names are nice.
# But put data in folder, and loop through the folder instead of the 

# image_one = getrgb('data/pic01.jpeg')
# print(image_one)

#Glob doenst read in order
# So your program should not rely on the sequence of the files

dataset=pandas.DataFrame()

# cd .. means go up one folder

for filepath in glob.glob('data/*'): # all the files in the folder # THE IMAGES ARE NOT IN ORDER !!
	image_features = pandas.DataFrame(getrgb(filepath))
	image_features = pandas.DataFrame.transpose(dataset)
	image_features['path'] = filepath #VERY IMPORTANT ! 
	pandas.concat([dataset, image_features])


#We dont want to overwrite, so we want to create a dataset. We need to transpose first, then call them image_features and then concat

#Apply GGM


gmm_dataset = dataset.iloc[:,0:3] # get column 0, 1, 2,
print(gmm_data) # to check 

gmm_data = preprocessing.normalize(gmm_data)
print(gmm_data)

gmm_machine = GaussianMixture(n_components = 4) # could try different numbers
gmm_machine.fit(gmm_data)
gmm_results = gmm_machine.predict(gmm_data)
print(results)

for n in (1,5):
	gmm_machine = GaussianMixture(n_components = n) # could try different numbers
	gmm_machine.fit(gmm_data)
	gmm_results = gmm_machine.predict(gmm_data)
	print(results)
		


#Results: 1 2 2 .... running 
# 2 0 3 3 ... running other time
#Why it is different? 

pyplot.scatter(dataset[0],dataset[1],c=gmm_results)
pyplot.savefig('scatter.png')
dataset['result'] = gmm_results

print(dataset)

dataset = dataset.sort_values(by=['path'])
