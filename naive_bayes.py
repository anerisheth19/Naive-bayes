#Applied AI - Homework 2
#Problem 3 - Use Naive Bayes on MNIST dataset 

#Written by Aneri Sheth - 801085402

#Import python libraries
from __future__ import print_function
from __future__ import division
from future.utils import iteritems
from builtins import range, input
import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal as distribution

#Function to read training and testing data
def read_data():
	#Process training data
	train_csv = pd.read_csv('mnist_train.csv',header=None)
	train_data = train_csv.values
	input_train = train_data[:,1:] / 256 #normalize input data
	class_train = train_data[:,0] #classes from 0 to 9

	#Process testing data
	test_csv = pd.read_csv('mnist_test.csv',header=None)
	test_data = test_csv.values
	input_test = test_data[:,1:] / 256 #normalize input data 
	class_test = test_data[:,0] #classes from 0 to 9
	
	return input_train, class_train, input_test, class_test

#Define a class for fitting gaussian into the data
class GaussianNB(object):
	#Function to get mean and variance of data 
	def gaussianFit(self, input_image, Y, smoothing = 0.0125):
		self.gaussians = dict()
		self.priors = dict()
		classes = set(Y)
		for i in classes:
			current_input = input_image[Y == i]
			self.gaussians[i] = { #use Gaussian dictionary 
				'mean': current_input.mean(axis=0), #computes mean of all class images in a loop
				'variance' : current_input.var(axis = 0) + smoothing, #computes variance 
			}
			self.priors[i] = float(len(Y[Y == i])) / len(Y) #get priors by gaussian


#Function to predict using gaussian probabilities
	def predict_test(self, input_image):
		N, D = input_image.shape
		K  = len(self.gaussians)
		prob = np.zeros((N,K)) #make an array for probabilities
		for i, j in iteritems(self.gaussians):
			mean, variance = j['mean'], j['variance']
			#use gaussian pdf to get the probabilities
			prob[:,i] = distribution.logpdf(input_image,mean = mean, cov = variance) + np.log(self.priors[i]) 

		return np.argmax(prob,axis=1) #gives the maximum values to predict the class label


#Function to get the overall testing data accuracy
	def total_accuracy(self, input_image, Y):
		prob = self.predict_test(input_image)
		#print(prob)
		return np.mean(prob == Y)

#Function to get accuracy per class label
	def accuracy_class(self,input_image,Y):
		prob = self.predict_test(input_image)
		classes = set(Y)
		count = 0
		for c in classes:
			current_input = Y[Y == 0] #accuracy for class 0 to 9 can be put here
		for k in range(len(Y)):
			if Y[k] == 0:
				if 0 == prob[k]:
					count = count + 1
		return count/len(current_input)


#This is the main function
if __name__ == '__main__':
	input_train, class_train, input_test, class_test = read_data()
	nb = GaussianNB()
	nb.gaussianFit(input_train, class_train)
	
	print("Overall Accuracy of Gaussian Naive Bayes for MNIST Data set:" , nb.total_accuracy(input_test, class_test))
	
	print("Accuracy for Class Label 0:", nb.accuracy_class(input_test,class_test))
