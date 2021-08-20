import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class UnivariateLinearRegression:

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
	# and when the second column contains the y values. INPUT: data frame of samples.
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

		for i in range(self.iterations):		# This loop will iterate according to the number of iterations.
			y_hat = self.updated_weight*self.X + self.updated_bias		# "y_hat" is the predicted label.

			error = self.mse(self.y, y_hat)		# Calculating the error of a single iteration.
			self.errors.append(error)			# Storing the error value to "errors" list.

			self.d_weight = (-2/self.n_samples) * np.sum(self.X * (self.y-y_hat))		# Derivative of MSE with the respect to weight.
			self.d_bias = (-2/self.n_samples) * np.sum(self.y - y_hat)					# Derivative of MSE with the respect to bias.

			self.updated_weight = self.updated_weight - (self.learning_rate*self.d_weight)		# Gradient descent. Updating "updated_weight".
			self.updated_bias = self.updated_bias - (self.learning_rate*self.d_bias)			# Gradient descent. Updating "updated_bias".

		self.errors = np.asarray(self.errors)		# Converting "errors" list into Numpy array.

	# Function to plot regression line before training.
	def plot_before(self, line_color='red', xlabel=None, ylabel=None, grid=False):
		plt.scatter(self.X, self.y)

		x_line = np.linspace(self.X.min(), self.X.max(), 100)		# Creating 100 values between the lowest X (leftmost) and largest X (rightmost).
		y_line = self.weight*x_line + self.bias						# Put every single value in "x_line" to the linear equation. This is the old weight and bias.
		plt.plot(x_line, y_line, c=line_color)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		if grid:
			plt.grid()		# Displaying grid (optional).
		plt.show()

	# Function to plot regression line after training.
	def plot_after(self, line_color='red', xlabel=None, ylabel=None, grid=False):
		plt.scatter(self.X, self.y)
		
		x_line = np.linspace(self.X.min(), self.X.max(), 100)		# Creating 100 values between the lowest X (leftmost) and largest X (rightmost).
		y_line = self.updated_weight*x_line + self.updated_bias	# Put every single value in "x_line" to the linear equation. This is the new weight and bias.
		plt.plot(x_line, y_line, line_color)
		plt.xlabel(xlabel)
		plt.ylabel(ylabel)
		if grid:
			plt.grid()		# Displaying grid (optional).
		plt.show()

	# Function to plot training progress (error value decrease).
	def plot_errors(self, grid=False):
		plt.plot(self.errors)
		plt.xlabel('Iterations')
		plt.ylabel('Error')
		if grid:
			plt.grid()		# Displaying grid (optional).
		plt.show()
