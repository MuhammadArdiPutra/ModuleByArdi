import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from tqdm import tqdm		# For displaying progress bar.



# DESCRIPTION: This is the implementation of Univariate Linear Regression model. 
# This function is compatible with data that lies in 1-dimensional space.
# INPUT: Refer to the __init__() function below.
class Linear_Regression_1D:

	# DESCRIPTION: Function to initialize the Perceptron model.
	# INPUTS: learning_rate=Determines training speed, iterations=Determines the number of iterations, random_state=Seed to lock randomness when initializing weight and bias.
	def __init__(self, learning_rate=0.001, iterations=100, random_state=None):

		np.random.seed(random_state)
		self.weight = np.random.uniform(-10, 10)		# Weight initialization. Random value between -10 and 10.
		self.bias = np.random.uniform(-10, 10)			# Bias initialization. Random value between -10 and 10.

		self.learning_rate = learning_rate				# Learning rate initialization.
		self.iterations = iterations					# Initialization of the number of iterations.

	# DESCRIPTION: Use this for loading data if the X and y are already separated from each other.
	# INPUTS: X=Feature (1-dimensional array), y=Labels.
	def take_data_raw(self, X, y):
		self.X = X							# Independent variable (feature).
		self.y = y							# Dependent variable (target).

		self.n_samples = len(self.X)		# Number of samples.
	
	# DESCRIPTION: Use this only when the first column of the csv file is the X and when the second column contains the y values.
	# INPUT: Pandas data frame, where column 0 is X and column 1 is y.
	def take_data_csv(self, csv_file):
		df = pd.read_csv(csv_file)
		self.X = df.iloc[:,0]		# Independent variable (feature).
		self.y = df.iloc[:,1]		# Dependent variable (target).

		self.n_samples = len(self.X)		# Number of samples.

	# DESCRIPTION: Function to calculate MSE. 
	# INPUT: y=actual label, y_hat=predicted label.
	# OUTPUT: error/loss value.
	def mse(self, y, y_hat):
		return np.sum((y-y_hat)**2)/self.n_samples		# MSE (Mean Squared Error) formula.

	# DESCRIPTION: Function to start training.
	# INPUT: No input.
	def train(self):
		
		self.updated_weight = self.weight			# "weight" contains the old weight. We will work with "updated_weight" starting from now on.
		self.updated_bias = self.bias				# "bias" contains the old weight. We will work with "updated_bias" starting from now on.

		self.errors = []							# Empty list to store the error of each iteration.

		for i in tqdm(range(self.iterations)):		# This loop will iterate according to the number of iterations.
			y_hat = self.updated_weight*self.X + self.updated_bias		# "y_hat" is the predicted label.

			error = self.mse(self.y, y_hat)			# Calculating the error of a single iteration.
			self.errors.append(error)				# Storing the error value to "errors" list.

			d_weight = (-2/self.n_samples) * np.sum(self.X * (self.y-y_hat))		# Derivative of MSE with the respect to weight (gradient).
			d_bias = (-2/self.n_samples) * np.sum(self.y - y_hat)					# Derivative of MSE with the respect to bias (gradient).

			self.updated_weight = self.updated_weight - (self.learning_rate*d_weight)		# Gradient descent. Updating "updated_weight".
			self.updated_bias = self.updated_bias - (self.learning_rate*d_bias)				# Gradient descent. Updating "updated_bias".

		self.errors = np.asarray(self.errors)		# Converting "errors" list into Numpy array.

	# DESCRIPTION: Use this function to predict single sample.
	# INPUT: X_test=A single sample to be predicted.
	def predict_single_sample(self, X_test):
		return self.updated_weight*X_test + self.updated_bias		# A linear equation with updated weight and bias.

	# DESCRIPTION: Use this function to predict multiple samples.
	# INPUT: X_test=Samples to be predicted in form of an array.
	def predict_multiple_samples(self, X_test):
		predictions = []				# Allocating empty list to store predictions later on.
		for sample in X_test:			# Iterate over each sample.
			prediction = self.predict_single_sample(sample)		# Using "predict_single_sample()" function to make a single prediction.
			predictions.append(prediction)
		
		return np.asarray(predictions)

	# DESCRIPTION: Function to plot regression line before training.
	# INPUT: line_color=As the name suggests, xlabel=Label for the X axis, ylabel=Label for the Y axis.
	def visualize_before(self, line_color='red', xlabel=None, ylabel=None):
		plt.scatter(self.X, self.y)

		x_line = np.linspace(self.X.min(), self.X.max(), 100)		# Creating 100 values between the lowest X (leftmost) and largest X (rightmost).
		y_line = self.weight*x_line + self.bias						# Put every single value in "x_line" to the linear equation. This is the old weight and bias.
		plt.plot(x_line, y_line, c=line_color)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

	# DESCRIPTION: Function to plot regression line after training.
	# INPUT: line_color=As the name suggests, xlabel=Label for the X axis, ylabel=Label for the Y axis.
	def visualize_after(self, line_color='red', xlabel=None, ylabel=None):
		plt.scatter(self.X, self.y)
		
		x_line = np.linspace(self.X.min(), self.X.max(), 100)		# Creating 100 values between the lowest X (leftmost) and largest X (rightmost).
		y_line = self.updated_weight*x_line + self.updated_bias	# Put every single value in "x_line" to the linear equation. This is the new weight and bias.
		plt.plot(x_line, y_line, line_color)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

	# DESCRIPTION: Function to plot training progress (error value decrease).
	# INPUT: print_details=Determines whether to print out the error before and after training.
	def plot_errors(self, print_details=False):
		plt.title('Training progress')
		plt.plot(self.errors)
		plt.xlabel('Iterations')
		plt.ylabel('Error')
		plt.show()

		if print_details:
			print('Initial error\t: {}'.format(self.errors[0]))
			print('Final error\t: {}'.format(self.errors[-1]))
		
		return



# DESCRIPTION: This is the implementation of Univariate Logistic Regression model. 
# This function is compatible with data that lies in 1-dimensional space.
# INPUT: Refer to the __init__() function below.
class Logistic_Regression_1D:

	# DESCRIPTION: Function to initialize the Logistic Regression model.
	# INPUTS: learning_rate=Determines training speed, iterations=Determines the number of iterations, random_state=Seed to lock randomness when initializing weight and bias.
	def __init__(self, learning_rate=0.001, iterations=100, random_state=None):
		
		np.random.seed(random_state)					# Setting a seed for locking randomness if the "random_state" is not set to None.
		self.weight = np.random.uniform(-10, 10)		# Weight initialization. Random value between -10 and 10.
		self.bias = np.random.uniform(-10, 10)			# Bias initialization. Random value between -10 and 10.

		self.learning_rate = learning_rate				# Learning rate initialization.
		self.iterations = iterations					# Initialization of the number of iterations.

	# DESCRIPTION: The sigmoid function.
	# INPUT: z=Value to be squeezed to the range of between 0 and 1.
	# OUTPUT: Value between 0 and 1.
	def sigmoid(self, z):
		return 1/(1+np.e**(-z))				# This is equivalent to 1/(1 + np.exp(-z)).

	# DESCRIPTION: Use this for loading data if the X and y are already separated from each other.
	# INPUTS: X=Feature (1-dimensional array), y=Labels.
	def take_data_raw(self, X, y):
		self.X = X							# Independent variable (feature).
		self.y = y							# Dependent variable (target).

		self.n_samples = len(self.X)		# Number of samples.
	
	# DESCRIPTION: Use this for loading data only when the first column of the csv file is the X
	# and when the second column contains the y values. 
	# INPUT: Pandas data frame of two columns (feature and target respectively).
	def take_data_csv(self, csv_file):
		df = pd.read_csv(csv_file)
		self.X = df.iloc[:,0]		# Independent variable (feature).
		self.y = df.iloc[:,1]		# Dependent variable (target).

		self.n_samples = len(self.X)		# Number of samples.

	# DESCRIPTION: The implementation of binary cross entropy loss function.
	# INPUTS: y=Actual label, y_hat=predicted label)
	# OUTPUT: loss=Loss/error value.
	def binary_crossentropy(self, y, y_hat):
		term_1 = y*np.log(y_hat)
		term_2 = (1-y) * np.log(1-y_hat)

		loss = -(np.sum(term_1 + term_2) / self.n_samples)		# THE DENOMINATOR SHOULD NOT BE HERE!!!!
		return loss
	
	# DESCRIPTION: Function to start training process.
	# INPUT: No input.
	def train(self):
		self.updated_weight = self.weight		# "weight" contains the old weight. We will work with "updated_weight" starting from now on.
		self.updated_bias = self.bias			# "bias" contains the old weight. We will work with "updated_bias" starting from now on.

		self.errors = []		# Empty list to store the error of each iteration.

		for _ in tqdm(range(self.iterations)):	# A progress bar will be displayed thanks to tqdm() function.
			y_hat = self.sigmoid(self.X*self.updated_weight + self.updated_bias)
			error = self.binary_crossentropy(self.y, y_hat)		# Calculating the error of each iteration.
			self.errors.append(error)		# Storing the error of each iteration.

			d_weight = np.sum(self.X *(y_hat-self.y)) / self.n_samples		# Derivative of Binary Crossentropy with the respect to weight (gradient).
			d_bias = np.sum(y_hat-self.y) / self.n_samples					# Derivative of Binary Crossentropy with the respect to bias (gradient).

			self.updated_weight = self.updated_weight - self.learning_rate*d_weight		# Gradient descent. Updating "updated_weight".
			self.updated_bias = self.updated_bias - self.learning_rate*d_bias			# Gradient descent. Updating "updated_bias".
		
		self.errors = np.asarray(self.errors)		# Converting "errors" list into Numpy array.

	# DESCRIPTION: Use this function to predict single sample.
	# INPUT: X_test=Samples to be predicted (single sample only).
	# OUTPUT: prediction=The predicted label (rounded).
	def predict_single_sample(self, X_test):
		return np.round(self.sigmoid(self.updated_weight*X_test + self.updated_bias))

	# DESCRIPTION: Use this function to predict multiple samples. 
	# INPUT: X_test=List of samples to be predicted.
	# OUTPUT: predictions=List of predicted labels.
	def predict_multiple_samples(self, X_test):
		predictions = []				# Allocating empty list to store predictions.
		for sample in X_test:			# Iterate through all testing samples.
			prediction = self.predict_single_sample(sample)		# Using "predict_single_sample()" function to make a single prediction.
			predictions.append(prediction)
		
		return np.asarray(predictions)

	# DESCRIPTION: Function to plot regression line before training.
	# INPUTS: x_start=Is an integer where should the starting x be (leftmost point), x_stop=Is the rightmost point, line_color=Color of the decision boundary, xlabel=Label for the X axis, ylabel=Label for the Y axis.
	def visualize_before(self, x_start=None, x_stop=None, line_color='red', xlabel=None, ylabel=None):
		plt.scatter(self.X, self.y, c=self.y, cmap='winter')

		if x_start == None:
			x_start = self.X.min()		# If "x_start" parameter is not provided, then use the minimum X.

		if x_stop == None:
			x_stop = self.X.max()		# If "x_stop" parameter is not provided, then use the maximum X.

		x_line = np.linspace(x_start, x_stop, 300)		# Creating 100 values between the lowest X (leftmost) and largest X (rightmost).
		y_line = self.sigmoid(self.weight*x_line + self.bias)						# Put every single value in "x_line" to the linear equation. This is the old weight and bias.
		plt.plot(x_line, y_line, c=line_color)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

	# DESCRIPTION: Function to plot regression line after training.
	# INPUTS: x_start=Is an integer where should the starting x be (leftmost point), x_stop=Is the rightmost point, line_color=Color of the decision boundary, xlabel=Label for the X axis, ylabel=Label for the Y axis.
	def visualize_after(self, x_start=None, x_stop=None, line_color='red', xlabel=None, ylabel=None):
		plt.scatter(self.X, self.y, c=self.y, cmap='winter')

		if x_start == None:
			x_start = self.X.min()		# If "x_start" parameter is not provided, then use the minimum X.

		if x_stop == None:
			x_stop = self.X.max()		# If "x_stop" parameter is not provided, then use the maximum X.
		
		x_line = np.linspace(x_start, x_stop, 300)		# Creating 100 values between the lowest X (leftmost) and largest X (rightmost).
		y_line = self.sigmoid(self.updated_weight*x_line + self.updated_bias)	# Put every single value in "x_line" to the linear equation. This is the new weight and bias.
		plt.plot(x_line, y_line, line_color)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

	# DESCRIPTIOn: Function to plot training progress (error value decrease).
	# INPUT: print_details=Determines whether or not to print out initial and final error values.
	def plot_errors(self, print_details=False):
		plt.title('Training progress')
		plt.plot(self.errors)
		plt.xlabel('Iterations')
		plt.ylabel('Error')
		plt.show()

		if print_details:
			print('Initial error\t: {}'.format(self.errors[0]))
			print('Final error\t: {}'.format(self.errors[-1]))



# DESCRIPTION: This is the implementation of categorical Perceptron model. 
# This function is compatible with data that lies in 2-dimensional space (as the name suggests).
# INPUT: Refer to the __init__() function below.
class Perceptron_2D:

	# DESCRIPTION: Function to initialize the Perceptron model.
	# INPUTS: learning_rate=Determines training speed, iterations=Determines the number of iterations, random_state=Seed to lock randomness when initializing weight and bias.
	def __init__(self, learning_rate=0.001, iterations=100, random_state=None):

		np.random.seed(random_state)			# Setting a seed for locking randomness if the "random_state" is not set to None.
		self.weight_0 = np.random.random()		# Initializing weight_0, weight_1, and bias.
		self.weight_1 = np.random.random()		# Since this function takes 2 input features, hence we need 2 weights.
		self.bias = np.random.random()			# Initializing bias.

		self.learning_rate = learning_rate		# Learning rate initialization.
		self.iterations = iterations			# Initialization of the number of iterations.
	
	# DESCRIPTION: We will use sigmoid for the activation function.
	# INPUTS: x=Value to be squeezed to the range of between 0 and 1.
	# OUTPUT: Value between 0 and 1.
	def sigmoid(self, x):
		return 1/(1 + np.exp(-x))		# This is equivalent to 1/(1+np.e**(-x)).
	
	# DESCRIPTION: Dereivative of the sigmoid activation function (for backpropagation).
	# INPUT: x=Value to be mapped using the derivative of sigmoid function.
	# OUTPUT: The output of the derivative of sigmoid funciton.
	def d_sigmoid(self, x):
		return self.sigmoid(x) * (1-self.sigmoid(x))

	# DESCRIPTION: Use this function when the features are already separated with the labels.
	# INPUTS: x=Features, y=Labels.
	def take_data_raw(self, X, y):
		self.X = np.asarray(X)					# Features initialization.
		self.y = np.asarray(y)					# Labels intialziation.

		self.n_samples  = self.X.shape[0]		# Taking the shape of first axis of X (number of samples).
		self.n_features = self.X.shape[1]		# Taking the shape of second axis of X (number of features).

	# DESCRIPTION: Function to take data directly from CSV file.
	# INPUTS: csv_file=Pandas data frame of 3 columns, where column 0, 1, and 2 are feature_0, feature_1, and target respectively.
	def take_data_csv(self, csv_file):
		df = pd.read_csv(csv_file)
		self.X = df.iloc[:,:2]		# Independent variables (features). Here we got 2 features.
		self.y = df.iloc[:,2]		# Dependent variable (target).

		self.n_samples  = self.X.shape[0]		# Number of samples.
		self.n_features = self.X.shape[1]		# Number of features.

	# DESCRIPTION: Function to calculate RSS (Residual Sum of Squares) error.
	# Note that this function is meant for calculating the error of a single sample (instead of multiple samples at once).
	# INPUTS: y=Actual label, y_hat=Predicted label.
	# OUTPUT: The error value.
	def rss(self, y, y_hat):
		return np.square(y_hat-y)

	# DESCRIPTION: RSS derivative (used for backpropagation).
	# INPUTS: y=Actual label, y_hat=Predicted label.
	# OUTPUT: A specific value obtained from the derivative of RSS function.
	def d_rss(self, y, y_hat):
		return 2*(y_hat-y)
	
	# DESCRIPTION: Function to start training process.
	# INPUT: No input.
	def train(self):
		
		self.updated_weight_0 = self.weight_0		# "weight_0" contains the old weight. We will work with "updated_weight_0" starting from now on.
		self.updated_weight_1 = self.weight_1		# "weight_1" contains the old weight. We will work with "updated_weight_0" starting from now on.
		self.updated_bias = self.bias				# "bias" contains the old weight. We will work with "updated_bias" starting from now on.

		self.errors = []		# An empty list to store errors of each iteration.
		for _ in tqdm(range(self.iterations)):		# This loop iterates according to the number of iterations.
			error_sum = 0
			for i in range(self.n_samples):			# This loop iterates acording to the number of samples (accessing every single sample one by one).
				
				# Forward propagation.
				z = self.X[i,0] * self.updated_weight_0 + self.X[i,1] * self.updated_weight_1 + self.updated_bias
				y_hat = self.sigmoid(z)		# Current prediction is stored in "y_hat".

				# Calculate error.
				error = self.rss(self.y[i], y_hat)
				error_sum += error		# The error of a single sample is summed up. Thus at the end we will have the total error within a single iteration.

				# Derivative of RSS error function (for backpropagation).
				d_error = self.d_rss(self.y[i], y_hat)

				# Derivative of sigmoid activation function (for backpropagation).
				d_activation = self.d_sigmoid(z)

				# Derivative of linear equation with respect to w0, w1, and b (for backpropagation).
				d_weight_0 = self.X[i,0]
				d_weight_1 = self.X[i,1]
				d_bias = 1

				# Backward propagation (calculating gradient for weight and bias update).
				weight_0_update = d_error * d_activation * d_weight_0		# Finding the value to update weight_0.
				weight_1_update = d_error * d_activation * d_weight_1		# Finding the value to update weight_1.
				bias_update = d_error * d_activation * d_bias				# Finding the value to update bias.

				# Gradient descent.
				self.updated_weight_0 = self.updated_weight_0 - self.learning_rate * weight_0_update		# Updating weight_0.
				self.updated_weight_1 = self.updated_weight_1 - self.learning_rate * weight_1_update		# Updating weight_1.
				self.updated_bias = self.updated_bias - self.learning_rate * bias_update					# Updating bias.
			
			self.errors.append(error_sum)		# Saving the error of a single iteration.

	# DESCRIPTION: Use this function to predict single sample.
	# INPUT: X_test=Samples to be predicted (single sample only).
	# OUTPUT: prediction=The predicted label (rounded).
	def predict_single_sample(self, X_test):
		z = self.updated_weight_0*X_test[0] + self.updated_weight_1*X_test[1] + self.updated_bias		# This process is exactly the same as forward propagation.
		prediction = self.sigmoid(z)
		return np.round(prediction)

	# DESCRIPTION: Use this function to predict multiple samples. 
	# INPUT: X_test=List of samples to be predicted.
	# OUTPUT: predictions=List of predicted labels.
	def predict_multiple_samples(self, X_test):
		predictions = []			# Allocating empty list to hold predictions.
		for sample in X_test:		# Predicting all test samples (might also be the training data itself).
			prediction = self.predict_single_sample(sample)		# Using "predict_single_sample()" function to make a single prediction.
			predictions.append(prediction)
		
		return np.asarray(predictions)

	# DESCRIPTION: Function to plot the data distribution along with the line before training.
	# INPUTS: line_color=Color of the decision boundary, xlabel=Label for the X axis, ylabel=Label for the Y axis.
	def visualize_before(self, line_color='red', xlabel=None, ylabel=None):
		m = -(self.bias/self.weight_1) / (self.bias/self.weight_0)		# Finding out m (slope in a linear equation).
		c = -self.bias/self.weight_1									# Finding out c (constant in a linear equation).

		plt.scatter(self.X[:,0], self.X[:,1], c=self.y, cmap='winter')		# Displaying the datapoints.

		x_line = np.linspace(self.X[:,0].min(), self.X[:,0].max(), 300)		# Creating 100 values between the lowest X (leftmost) and largest X (rightmost).
		y_line = m*x_line + c						# Put every single value in "x_line" to the linear equation. This is the old weight and bias.
		plt.plot(x_line, y_line, c=line_color)		# Plotting the decision boundary.
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

	# DESCRIPTION: Function to plot the data distribution along with the line after training.
	# INPUTS: line_color=Color of the decision boundary, xlabel=Label for the X axis, ylabel=Label for the Y axis.
	def visualize_after(self, line_color='red', xlabel=None, ylabel=None):
		m = -(self.updated_bias/self.updated_weight_1) / (self.updated_bias/self.updated_weight_0)		# Finding out m (slope in a linear equation).
		c = -self.updated_bias/self.updated_weight_1									# Finding out c (constant in a linear equation).

		plt.scatter(self.X[:,0], self.X[:,1], c=self.y, cmap='winter')		# Displaying the datapoints.

		x_line = np.linspace(self.X[:,0].min(), self.X[:,0].max(), 300)		# Creating 100 values between the lowest X (leftmost) and largest X (rightmost).
		y_line = m*x_line + c						# Put every single value in "x_line" to the linear equation. This is the old weight and bias.
		plt.plot(x_line, y_line, c=line_color)		# Plotting the decision boundary.
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

	# DESCRIPTIOn: Function to plot training progress (error value decrease).
	# INPUT: print_details=Determines whether or not to print out initial and final error values.
	def plot_errors(self, print_details=False):
		plt.title('Training progress')
		plt.plot(self.errors)
		plt.xlabel('Iterations')
		plt.ylabel('Error')
		plt.show()

		if print_details:
			print('Initial error\t: {}'.format(self.errors[0]))
			print('Final error\t: {}'.format(self.errors[-1]))



# DESCRIPTION: This is the implementation of categorical Naive Bayes model. 
# This function is compatible with data that lies in multidimensional space.
# INPUT: Refer to the __init__() function below.
class Categorical_Naive_Bayes:
	
	# DESCRIPTION: Function to initialize the Naive Bayes model.
	# INPUTS: No input.
	def __init__(self):
		pass		# There is nothing to be done during the initialization since Naive Bayes does not require parameters like learning rate, weight, bias, K, etc.

	# DESCRIPTION: Function to take the data. Use this function if the features and labels are already separated.
	# INPUTS: X=Features, y=labels.
	def take_data_raw(self, X, y):
		self.X = np.asarray(X)		# Features initialization.
		self.y = np.asarray(y)		# Labels initialization.

		self.n_samples  = self.X.shape[0]		# Taking the shape of first axis of X (number of samples).
		self.n_features = self.X.shape[1]		# Taking the shape of second axis of X (number of features).

	# DESCRIPTION: Similar to take_data_raw(). Use this function if the features and labels are stored in a
	# single CSV file, where the first 2 columns should be the features and the third column is the label.
	# INPUTS: csv_file=Pandas data frame where column 0 and 1 is X and column 2 is y.
	def take_data_csv(self, csv_file):
		df = pd.read_csv(csv_file)
		self.X = df.iloc[:,:-1]		# Independent variables (features). Here we got 2 features.
		self.y = df.iloc[:,-1]		# Dependent variable (target).

		self.n_samples  = self.X.shape[0]		# Number of samples.
		self.n_features = self.X.shape[1]		# Number of features.

	# DESCRIPTION: Function to calculate the prior probability of all classes.
	# INPUTS: y=List of labels.
	# OUTPUTS: classes=Prior probability of all classes.
	def calculate_priors(self, y):
		unique = np.unique(y, return_counts=True)		# Finding out the number of unique classes and each of the counts.

		classes = [0] * len(unique[0])		# Number of unique classes.
		classes_count = unique[1]			# Number of samples in each class. (Please refer to the output of "np.unique()").

		for i in range(len(classes)):		# Iterate through all unique classes.
			classes[i] = classes_count[i]/np.sum(classes_count)		# Calculating the prior probability of each class. (This is based on the formula of prior probability).

		return np.asarray(classes)		# This is the prior probability of all available classes.

	# DESCRIPTION: This function is used to calculate the likelihood in order to be able to complete the following formula --> Posterior = Likelihood * Prior.
	# INPUTS: x_train_slice=Values of training samples taken from a single feature, y_train=Labels of datapoints in training set, current_x=A feature that we are currently working with, current_y=A label that we are currently working with.
	# OUTPUTS: likelihood=Value for the likelihood.
	def calculate_likelihood(self, X_train_slice, y_train, current_X, current_y):
		X_train_slice = X_train_slice[y_train==current_y]		# Take only a single feature which corresponds to the currently selected target. The result is stored in "X_train_slice".
																# This function will work for each label (will be called in a loop).
																
		count_X = len(X_train_slice)									# "count_X" is the number of samples of a specific label.
		count_current_X = len(np.where(X_train_slice==current_X)[0])	# "count_current_X" is the number of samples of a spscific label of a specific feature (number of samples that have the exact same feature).

		likelihood = count_current_X / count_X					# This is just the formula to calculate likelihood.
		return likelihood

	# DESCRIPTION: Function to predicte multiple samples that are stored in a single array.
	# INPUTS: X_train=Features of the training data, X_test=List of samples to be predicted, y_train=Labels of the training data, return_probability=Whether or not you want to take the probability value instead of the final prediction result.
	# OUTPUTS: predictions=This can be either the actual prediction or the probability value. Keep in mind that this is not the actual probability value. Instead, it is just a value which behaves similarly to the actual probability.
	def predict_multiple_samples(self, X_train, X_test, y_train, return_probability=False):
		predictions = []			# Allocating empty list to store predictions.

		for i in range(len(X_test)):					# Iterate through all testing samples.
			priors = self.calculate_priors(y_train)		# Finding out the prior probability of all classes.
			posteriors = [0] * len(priors)				# Let the initial posterior probability of all classes being 0.

			for j in range(len(posteriors)):			# Iterate through all possible labels.
				likelihood = 1							# Set the initial likelihood to 1 (we will multiply this with the likelihood values of each feature).

				for k in range(X_train.shape[1]):		# Iterate through all features.
					likelihood *= self.calculate_likelihood(X_train[:,k], y_train, X_test[:,k][i], j)		# Calculating the likelihood of each feature.
				
				posteriors[j] = likelihood * priors[j]	# Caklculate the posterior probability.

			predictions.append(posteriors)				# Store the posterior probabilities to "predictions" list.
		
		if return_probability:							# If "return_probability" is set to True, then the raw predictions are returned.
			return np.asarray(predictions)

		else:											# If "return_probability" is set to False, then the rounded predictions are returned.
			return np.argmax(np.asarray(predictions), axis=1)



# DESCRIPTION: This is the implementation of KNN (K-Nearest Neighbor) model. 
# This function is compatible with data that lies in multidimensional space (not limited to 2 dimensions only).
# INPUT: Refer to the __init__() function below.
class K_Nearest_Neighbors:

	# DESCRIPTION: Function to initialize the KNN model.
	# INPUTS: K=The number of nearest neighbor to take into account for predicting the label of test data.
	def __init__(self, K):
		self.K = K

	# DESCRIPTION: Function to take the data. Use this function if the features and labels are already separated.
	# INPUTS: X=Features, y=labels.
	def take_data_raw(self, X, y):
		self.X = X		# Features initialization.
		self.y = y		# Labels initialization.

		self.n_samples = len(self.X)		# Number of samples.
	
	# DESCRIPTION: Similar to take_data_raw(). Use this function if the features and labels are stored in a
	# single CSV file, where the rightmost column is the label and the rest are the features.
	# INPUTS: csv_file=Pandas data frame where the last column (rightmost) is the target, while the rest are the X.
	def take_data_csv(self, csv_file):
		df = pd.read_csv(csv_file)
		self.X = df.iloc[:,:-1].values		# Features initialization.
		self.y = df.iloc[:,-1].values		# Labels initialization.

		self.n_samples = len(self.X)		# Number of samples.

	# DESCRIPTION: Function to calculate the euclidean distance between a point to another.
	# INPUTS: x0 and x1 = the points that the distance will be calculated.
	def calculate_distance(self, x0, x1):
		return np.sqrt(np.sum((x0-x1)**2))		# Eucliedan distance formula.

	# DESCRIPTION: Function to predict multiple samples. 
	# INPUTS: X_test=List of datapoints to predict.
	# OUTPUTS: predictions=Predicted labels for each sample in X_test.
	def predict_multiple_samples(self, X_test):
		predictions = []					# Allocating empty list to store predictions.
		for i in range(len(X_test)):		# Iterate through all testing features.
			distances = []					# Allocating empty list to store distances between a point in testing list to all training points.
			for j in range(len(self.X)):	# Iterate through all training features.
				distance = self.calculate_distance(X_test[i], self.X[j])		# Calculate the distance between a point in testing list to a single point in trainig list.
				distances.append(distance)	# Store the distance to the "distances" list.

			sorted_indices = np.argsort(distances)		# Sort the "distances" list, but store the indices only. The first element of "sorted_indices" is the index containing the shortest distance.

			k_nearest_neighbors = sorted_indices[:self.K]		# Take only the indices of the several shortest distances. The word 'several' depends on the value of K.

			k_nearest_points = self.X[k_nearest_neighbors]				# Take the K points that are closest to the current testing data. (This line is not very necessary though).
			k_nearest_labels = self.y[k_nearest_neighbors].astype(int)	# Take the labels of the K points that are closest to the current testing data. The astype(int) function is used just to make np.bincount() working properly.

			most_common = np.bincount(k_nearest_labels).argmax()		# The voting process. Taking the most common K labels.

			predictions.append(most_common)				# Store prediction result to "predictions" list.

		return predictions

	# DESCRIPTION: Function to visualize training data distribution on a 2-dimensional space. This will only work properly if the data is only having 2 features.
	# INPUTS: No input.
	def visualize_train_only(self):
		plt.scatter(self.X[:,0], self.X[:,1], c=self.y, cmap='winter')		# Just a standard scatter plot using Matplotlib where the datapoints colors are determined by the corresponding labels.

	# DESCRIPTION: Function to visualize both training and testing data. This will only work properly if the data is only having 2 features.
	# INPUTS: X_test=Features of test data, predictions=Predicted labels of the testing data which is obtained from the predict_multiple_samples() function.
	def visualize_train_test(self, X_test, predictions):
		plt.scatter(self.X[:,0], self.X[:,1], c=self.y, cmap='winter')		# Scatter plot to display the training data.
		plt.scatter(X_test[:,0], X_test[:,1], c=predictions, marker='x', s=60, cmap='winter')		# Scatter plot to display testing data along with the predicted labels.



# DESCRIPTION: This is the implemnetation of the SVM model with linear kernel.
# INPUT: Refer to the __init__() function below.
class Linear_Support_Vector_Machine_2D:

	def __init__(self, learning_rate=0.001, iterations=100, lambda_=0.01, random_state=None):

		np.random.seed(random_state)
		self.weights = np.random.random(2)
		self.bias = np.random.random()

		self.learning_rate = learning_rate
		self.iterations = iterations
		self.lambda_ = lambda_

	# DESCRIPTION: Function to take the data. Use this function if the features and labels are already separated.
	# INPUTS: X=Features, y=labels.
	def take_data_raw(self, X, y):
		self.X = X		# Features initialization.
		self.y = y		# Labels initialization.
		self.convert_labels()		# Automatically convert all labels 0 to -1.

		self.n_samples = len(self.X)		# Number of samples.

	# DESCRIPTION: Similar to take_data_raw(). Use this function if the features and labels are stored in a
	# single CSV file, where the rightmost column is the label and the rest are the features.
	# INPUTS: csv_file=Pandas data frame where the last column (rightmost) is the target, while the rest are the X.
	def take_data_csv(self, csv_file):
		df = pd.read_csv(csv_file)
		self.X = df.iloc[:,:-1].values		# Features initialization.
		self.y = df.iloc[:,-1].values		# Labels initialization.
		self.convert_labels()				# Automatically convert all labels 0 to -1.

		self.n_samples = len(self.X)		# Number of samples.

	# DESCRIPTION: Function to convert labels 0 to -1. This is done because SVM works by classifying either 1 or -1 instead of 1 or 0.
	# INPUT: No input.
	def convert_labels(self):
		self.y = np.where(self.y==0, -1, 1)		# For all y, IF y is 0, then convert to -1, else don't convert (remain 1).

	# DESCRIPTION: The equation of the decision boundary.
	# INPUT: x_i=Current sample, y_i=Current label, w=Weights, b=Bias.
	def linear_equation(self, x_i, y_i, w, b):
		return y_i*((np.dot(x_i, w)) - b)

	# DESCRIPTION: The hinge loss function.
	# INPUT: X_i=Current sample, y_i=Current label, w=Weights, b=Bias.
	def hinge_loss(self, X_i, y_i):
		return np.max([0, 1 - (y_i * (np.dot(X_i, self.updated_weights) - self.updated_bias))])
	
	# DESCRIPTION: The combination of hinge loss and its regularization term.
	# INPUT: X_i=Current sample, y_i=Current label, w=Weights, b=Bias, lambda_=Weighting for the regularization term, condition=Will be True if the linear equation result is >= 1.
	def hinge_reg_loss(self, X_i, y_i, condition):
		hinge = 1 - (y_i * (np.dot(X_i, self.updated_weights)) - self.updated_bias)					# The hinge loss formula.
		reg = self.lambda_*(np.sqrt(np.dot(self.updated_weights, self.updated_weights)))			# The regularization term.

		if condition:		# If the condition is True, then we will only use regularization part.
			return reg
		else:				# Else both regularization and the hinge loss itself is going to be taken into account.
			return reg + hinge

	# DESCRIPTION: Function to start training process.
	# INPUT: No input.
	def train(self):
		self.updated_weights = self.weights.astype(float)		# Starting from now on we are going to work with the self.updated_weights and self.updated_bias so that we can compare the old and new weights.
		self.updated_bias = np.float(self.bias)

		self.losses = []					# Error values without lambda.
		self.losses_hinge_reg = []			# Error values with regularization (lambda)

		for i in tqdm(range(self.iterations)):		# Iterate according to the number of iterations.

			loss_sum = 0					# For summing the loss without regularization.
			loss_hinge_reg_sum = 0			# For summing the loss with regularization.

			for j in range(self.n_samples):		# Iterate through all samples.
				condition = self.linear_equation(self.X[j], self.y[j], self.updated_weights, self.updated_bias) >= 1		# This will be true if the result of linear equation is >= 1.
				
				loss_sum += self.hinge_loss(self.X[j], self.y[j])								# Add the current loss to loss_sum. At the end of the iteration we are going to obtain the sum of all loss of each datapoint.
				loss_hinge_reg_sum += self.hinge_reg_loss(self.X[j], self.y[j], condition)		# Similar to the previous line, but here we also take into account the regularization term.

				if condition:
					self.updated_weights -= self.learning_rate * (2*self.lambda_*self.updated_weights)		# This is how the weight being updated when the condition is True. Bias is not updated at this part.
				else:
					self.updated_weights -= self.learning_rate * ((2*self.lambda_*self.updated_weights) - np.dot(self.X[j],self.y[j]))		# This is how the weight being updated when the condition is False. Bias is updated as well.
					self.updated_bias -= self.learning_rate * self.y[j]
				
			loss_sum = loss_sum / self.n_samples						# Dividing the errors with the number of samples (kinda like the average loss of each datapoints).
			loss_hinge_reg_sum = loss_hinge_reg_sum / self.n_samples	# Dividing the errors with the number of samples (kinda like the average loss of each datapoints).
			
			self.losses.append(loss_sum)			# Keeping track the loss without regularization.
			self.losses_hinge_reg.append(loss_hinge_reg_sum)		# Keeping track the loss with regularization.

	# DESCRIPTION: Function to make predictions
	# INPUT: X_test=Features that the labels are going to be predicted, weights=Trained weights, bias=Trained bias.
	def predict_multiple_samples(self, X_test):
		predictions = np.dot(X_test, self.updated_weights) - self.updated_bias
		return np.sign(predictions)			# All negative values are going to be converted to -1, positive values are converted to 1.

	# DESCRIPTIOn: Function to plot training progress (error value decrease).
	# INPUT: print_details=Determines whether or not to print out initial and final error values.
	def plot_errors(self, print_details=False):
		plt.title('Training progress')
		plt.plot(self.losses_hinge_reg)
		plt.xlabel('Iterations')
		plt.ylabel('Error')
		plt.show()

		if print_details:
			print('Initial error\t: {}'.format(self.losses_hinge_reg[0]))
			print('Final error\t: {}'.format(self.losses_hinge_reg[-1]))

	# DESCRIPTION: This is used for the visualization purpose.
	# INPUT: x=features, w=weights, b=bias, offset=determines whether taking the negative or positive side of the decision boundary.
	def get_hyperplane_value(self, x, w, b, offset):
		return (-w[0] * x + b + offset) / w[1]

	# DECRIPTION: Function to display datapoints along with the hyperplane (decision boundary) before training.
	# INPUT: No input.
	def visualize_before(self):
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		plt.scatter(self.X[:,0], self.X[:,1], c=self.y, cmap='winter')
		
		x0_1 = np.amin(self.X[:, 0])
		x0_2 = np.amax(self.X[:, 0])
		
		x1_1 = self.get_hyperplane_value(x0_1, self.weights, self.bias, 0)			# Determining the decision boundary position.
		x1_2 = self.get_hyperplane_value(x0_2, self.weights, self.bias, 0)
		
		x1_1_m = self.get_hyperplane_value(x0_1, self.weights, self.bias, -1)		# Determining the margin position at negative side.
		x1_2_m = self.get_hyperplane_value(x0_2, self.weights, self.bias, -1)
		
		x1_1_p = self.get_hyperplane_value(x0_1, self.weights, self.bias, 1)		# Determining the margin position at positive side.
		x1_2_p = self.get_hyperplane_value(x0_2, self.weights, self.bias, 1)
		
		ax.plot([x0_1, x0_2], [x1_1, x1_2], "k--")			# The decision boundary.
		ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "r")		# Margin at the negative side.
		ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "r")		# Margin at the positive side.
		
		x1_min = np.amin(self.X[:, 1])
		x1_max = np.amax(self.X[:, 1])
		ax.set_ylim([x1_min - 3, x1_max + 3])				# Determining maximum and minimum value in the scatter plot.
		
		plt.show()

	# DECRIPTION: Function to display datapoints along with the hyperplane (decision boundary) after training.
	# INPUT: No input.
	def visualize_after(self):
		fig = plt.figure()
		ax = fig.add_subplot(1, 1, 1)
		plt.scatter(self.X[:,0], self.X[:,1], c=self.y, cmap='winter')
		
		x0_1 = np.amin(self.X[:, 0])
		x0_2 = np.amax(self.X[:, 0])
		
		x1_1 = self.get_hyperplane_value(x0_1, self.updated_weights, self.updated_bias, 0)			# Determining the decision boundary position.
		x1_2 = self.get_hyperplane_value(x0_2, self.updated_weights, self.updated_bias, 0)
	
		x1_1_m = self.get_hyperplane_value(x0_1, self.updated_weights, self.updated_bias, -1)		# Determining the margin position at negative side.
		x1_2_m = self.get_hyperplane_value(x0_2, self.updated_weights, self.updated_bias, -1)
		
		x1_1_p = self.get_hyperplane_value(x0_1, self.updated_weights, self.updated_bias, 1)		# Determining the margin position at positive side.
		x1_2_p = self.get_hyperplane_value(x0_2, self.updated_weights, self.updated_bias, 1)
		
		ax.plot([x0_1, x0_2], [x1_1, x1_2], "k--")			# The decision boundary.
		ax.plot([x0_1, x0_2], [x1_1_m, x1_2_m], "r")		# Margin at the negative side.
		ax.plot([x0_1, x0_2], [x1_1_p, x1_2_p], "r")		# Margin at the positive side.
		
		x1_min = np.amin(self.X[:, 1])
		x1_max = np.amax(self.X[:, 1])
		ax.set_ylim([x1_min - 3, x1_max + 3])		# Determining maximum and minimum value in the scatter plot.
		
		plt.show()