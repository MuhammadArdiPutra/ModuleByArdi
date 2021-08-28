import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from tqdm import tqdm		# For displaying progress bar.

class Univariate_Linear_Regression:

	def __init__(self, learning_rate=0.001, iterations=100, random_state=None):

		np.random.seed(random_state)
		self.weight = np.random.uniform(-10, 10)		# Weight initialization. Random value between -10 and 10.
		self.bias = np.random.uniform(-10, 10)			# Bias initialization. Random value between -10 and 10.
		self.learning_rate = learning_rate
		self.iterations = iterations

	# Use this if the X and y are already separated from each other.
	def take_data_raw(self, X, y):
		self.X = X		# Independent variable (feature).
		self.y = y		# Dependent variable (target).

		self.n_samples = len(self.X)		# Number of samples.
	
	# Use this only when the first column of the csv file is the X
	# and when the second column contains the y values. INPUT: Pandas data frame, where column 0 is X and column 1 is y.
	def take_data_csv(self, csv_file):
		df = pd.read_csv(csv_file)
		self.X = df.iloc[:,0]		# Independent variable (feature).
		self.y = df.iloc[:,1]		# Dependent variable (target).

		self.n_samples = len(self.X)		# Number of samples.

	# Function to calculate MSE. INPUT: (actual label, predicted label). OUTPUT: error/loss value.
	def mse(self, y, y_hat):		# MSE (Mean Squared Error).
		return np.sum((y-y_hat)**2)

	# Function to start training.
	def train(self):
		
		self.updated_weight = self.weight		# "weight" contains the old weight. We will work with "updated_weight" starting from now on.
		self.updated_bias = self.bias			# "bias" contains the old weight. We will work with "updated_bias" starting from now on.

		self.errors = []		# Empty list to store the error of each iteration.

		for i in tqdm(range(self.iterations)):		# This loop will iterate according to the number of iterations.
			y_hat = self.updated_weight*self.X + self.updated_bias		# "y_hat" is the predicted label.

			error = self.mse(self.y, y_hat)		# Calculating the error of a single iteration.
			self.errors.append(error)			# Storing the error value to "errors" list.

			d_weight = (-2/self.n_samples) * np.sum(self.X * (self.y-y_hat))		# Derivative of MSE with the respect to weight (gradient).
			d_bias = (-2/self.n_samples) * np.sum(self.y - y_hat)					# Derivative of MSE with the respect to bias (gradient).

			self.updated_weight = self.updated_weight - (self.learning_rate*d_weight)		# Gradient descent. Updating "updated_weight".
			self.updated_bias = self.updated_bias - (self.learning_rate*d_bias)			# Gradient descent. Updating "updated_bias".

		self.errors = np.asarray(self.errors)		# Converting "errors" list into Numpy array.

	# Use this function to predict single sample.
	def predict_single_sample(self, X_test):
		return self.updated_weight*X_test + self.updated_bias		# A linear equation with updated weight and bias.

	# Use this function to predict multiple samples. INPUT: "X_test" is an array.
	def predict_multiple_samples(self, X_test):
		predictions = []
		for sample in X_test:			# Iterate over each sample.
			prediction = self.predict_single_sample(sample)		# Using "predict_single_sample()" function to make a single prediction.
			predictions.append(prediction)
		
		return np.asarray(predictions)

	# Function to plot regression line before training.
	def plot_before(self, line_color='red', xlabel=None, ylabel=None):
		plt.scatter(self.X, self.y)

		x_line = np.linspace(self.X.min(), self.X.max(), 100)		# Creating 100 values between the lowest X (leftmost) and largest X (rightmost).
		y_line = self.weight*x_line + self.bias						# Put every single value in "x_line" to the linear equation. This is the old weight and bias.
		plt.plot(x_line, y_line, c=line_color)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

	# Function to plot regression line after training.
	def plot_after(self, line_color='red', xlabel=None, ylabel=None):
		plt.scatter(self.X, self.y)
		
		x_line = np.linspace(self.X.min(), self.X.max(), 100)		# Creating 100 values between the lowest X (leftmost) and largest X (rightmost).
		y_line = self.updated_weight*x_line + self.updated_bias	# Put every single value in "x_line" to the linear equation. This is the new weight and bias.
		plt.plot(x_line, y_line, line_color)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

	# Function to plot training progress (error value decrease).
	def plot_errors(self, print_details=False):
		plt.title('Training progress')
		plt.plot(self.errors)
		plt.xlabel('Iterations')
		plt.ylabel('Error')
		plt.show()

		if print_details:
			print('Initial error:\t{}'.format(self.errors[0]))
			print('Final error:\t{}'.format(self.errors[-1]))
		
		return

################################################################################################################

class Univariate_Logistic_Regression():

	def __init__(self, learning_rate=0.001, iterations=100, random_state=None):
		
		np.random.seed(random_state)
		self.weight = np.random.uniform(-10, 10)		# Weight initialization. Random value between -10 and 10.
		self.bias = np.random.uniform(-10, 10)			# Bias initialization. Random value between -10 and 10.
		self.learning_rate = learning_rate
		self.iterations = iterations

	# The sigmoid function.
	def sigmoid(self, z):
		return 1/(1+np.e**(-z))

	# Use this if the X and y are already separated from each other.
	def take_data_raw(self, X, y):
		self.X = X		# Independent variable (feature).
		self.y = y		# Dependent variable (target).

		self.n_samples = len(self.X)		# Number of samples.
	
	# Use this only when the first column of the csv file is the X
	# and when the second column contains the y values. INPUT: data frame of samples.
	def take_data_csv(self, csv_file):
		df = pd.read_csv(csv_file)
		self.X = df.iloc[:,0]		# Independent variable (feature).
		self.y = df.iloc[:,1]		# Dependent variable (target).

		self.n_samples = len(self.X)		# Number of samples.

	# INPUT: (actual label, predicted label). OUTPUT: error/loss value.
	def binary_crossentropy(self, y, y_hat):
		term_1 = y*np.log(y_hat)
		term_2 = (1-y) * np.log(1-y_hat)

		loss = -(np.sum(term_1 + term_2) / self.n_samples)		# THE DENOMINATOR SHOULD NOT BE HERE!!!!
		return loss
	
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

	# Use this function to predict single sample.
	def predict_single_sample(self, X_test):
		return np.round(self.sigmoid(self.updated_weight*X_test + self.updated_bias))

	# Use this function to predict multiple samples. "X_test" is an array.
	def predict_multiple_samples(self, X_test):
		predictions = []
		for sample in X_test:
			prediction = self.predict_single_sample(sample)		# Using "predict_single_sample()" function to make a single prediction.
			predictions.append(prediction)
		
		return np.round(np.asarray(predictions))

	# Function to plot regression line before training.
	# INPUT: "x_start" is an integer where should the starting x be (leftmost point). "x_stop" is simply the rightmost point.
	def plot_before(self, x_start=None, x_stop=None, line_color='red', xlabel=None, ylabel=None):
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

	# Function to plot regression line after training.
	# INPUT: "x_start" is an integer where should the starting x be (leftmost point). "x_stop" is simply the rightmost point.
	def plot_after(self, x_start=None, x_stop=None, line_color='red', xlabel=None, ylabel=None):
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

	def plot_errors(self, print_details=False):
		plt.title('Training progress')
		plt.plot(self.errors)
		plt.xlabel('Iterations')
		plt.ylabel('Error')
		plt.show()

		if print_details:
			print('Initial error:\t{}'.format(self.errors[0]))
			print('Final error:\t{}'.format(self.errors[-1]))
		
		return

################################################################################################################

class Perceptron_2D():
	
	def __init__(self, learning_rate=0.001, iterations=100, random_state=None):

		np.random.seed(random_state)
		self.weight_0 = np.random.random()		# Initializing weight_0, weight_1, and bias.
		self.weight_1 = np.random.random()		# Since this function takes 2 input features, hence we need 2 weights.
		self.bias = np.random.random()			# Initializing bias.

		self.learning_rate = learning_rate		# Learning rate initialization.
		self.iterations = iterations			# Initialization of the number of iterations.
	
	# We will use sigmoid for the activation function.
	def sigmoid(self, x):
		return 1/(1 + np.exp(-x))	
	
	# Dereivative of the sigmoid activation function (for backpropagation).
	def d_sigmoid(self, x):
		return self.sigmoid(x) * (1-self.sigmoid(x))

	# Use this function when the features are already separated with the labels.
	def take_data_raw(self, X, y):
		self.X = np.asarray(X)
		self.y = np.asarray(y)

		self.n_samples  = self.X.shape[0]		# Taking the shape of first axis of X (number of samples).
		self.n_features = self.X.shape[1]		# Taking the shape of second axis of X (number of features).

	# Function to take data directly from CSV file.
	# INPUT: Pandas data frame of 3 columns, where column 0, 1, and 2 are feature_0, feature_1, and target respectively.
	def take_data_csv(self, csv_file):
		df = pd.read_csv(csv_file)
		self.X = df.iloc[:,:2]		# Independent variables (features). Here we got 2 features.
		self.y = df.iloc[:,2]		# Dependent variable (target).

		self.n_samples  = self.X.shape[0]		# Number of samples.
		self.n_features = self.X.shape[1]		# Number of features.

	# Function to calculate RSS (Residual Sum of Squares) error.
	# Note that this function is meant for calculating the error of a single sample.
	def rss(self, y, y_hat):
		return np.square(y_hat-y)

	# RSS derivative (used for backpropagation).
	def d_rss(self, y, y_hat):
		return 2*(y_hat-y)
	
	# The training process goes below.
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

	# Use this function to predict single sample.
	def predict_single_sample(self, X_test):
		z = self.updated_weight_0*X_test[0] + self.updated_weight_1*X_test[1] + self.updated_bias		# This process is exactly the same as forward propagation.
		prediction = self.sigmoid(z)
		return np.round(prediction)

	# Use this function to predict multiple samples. "X_test" is an array.
	def predict_multiple_samples(self, X_test):
		predictions = []
		for sample in X_test:		# Predicting all test samples (might also be the training data itself).
			prediction = self.predict_single_sample(sample)		# Using "predict_single_sample()" function to make a single prediction.
			predictions.append(prediction)
		
		return np.asarray(predictions)

	# Function to plot the data distribution along with the line before training.
	def plot_before(self, line_color='red', xlabel=None, ylabel=None):
		m = -(self.bias/self.weight_1) / (self.bias/self.weight_0)		# Finding out m (slope in a linear equation).
		c = -self.bias/self.weight_1									# Finding out c (constant in a linear equation).

		plt.scatter(self.X[:,0], self.X[:,1], c=self.y, cmap='winter')		# Displaying the datapoints.

		x_line = np.linspace(self.X[:,0].min(), self.X[:,0].max(), 300)		# Creating 100 values between the lowest X (leftmost) and largest X (rightmost).
		y_line = m*x_line + c						# Put every single value in "x_line" to the linear equation. This is the old weight and bias.
		plt.plot(x_line, y_line, c=line_color)		# Plotting the decision boundary.
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

	# Function to plot the data distribution along with the line after training.
	def plot_after(self, line_color='red', xlabel=None, ylabel=None):
		m = -(self.updated_bias/self.updated_weight_1) / (self.updated_bias/self.updated_weight_0)		# Finding out m (slope in a linear equation).
		c = -self.updated_bias/self.updated_weight_1									# Finding out c (constant in a linear equation).

		plt.scatter(self.X[:,0], self.X[:,1], c=self.y, cmap='winter')		# Displaying the datapoints.

		x_line = np.linspace(self.X[:,0].min(), self.X[:,0].max(), 300)		# Creating 100 values between the lowest X (leftmost) and largest X (rightmost).
		y_line = m*x_line + c						# Put every single value in "x_line" to the linear equation. This is the old weight and bias.
		plt.plot(x_line, y_line, c=line_color)		# Plotting the decision boundary.
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		plt.show()

	# Function to plot training progress (error value decrease).
	def plot_errors(self, print_details=False):
		plt.title('Training progress')
		plt.plot(self.errors)
		plt.xlabel('Iterations')
		plt.ylabel('Error')
		plt.show()

		if print_details:
			print('Initial error:\t{}'.format(self.errors[0]))
			print('Final error:\t{}'.format(self.errors[-1]))
		
		return










