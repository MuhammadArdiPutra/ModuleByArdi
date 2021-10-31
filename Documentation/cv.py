import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# DESCRIPTION: The class below is the implementation of convolution process from scratch.
# INPUT: Refer to the __init__() function below.
class Convolution():

	# DESCRIPTION: This function will got called when a new Convolution object is intialized.
	# It is important to know that the kernel size is 3x3.
	# INPUT: kernel=Determines the kernel to be used. random_state=This will work only when the kernel is set to 'random'.
	def __init__(self, kernel='vertical_edge', random_state=None):

		if kernel == 'vertical_edge':				# This is the implementation of vertical edge detector.
			self.kernel = np.asarray([[-1, 0, 1], 
									  [-1, 0, 1], 
									  [-1, 0, 1]])
		elif kernel == 'horizontal_edge':			# This is the implementation of horizontal edge detector.
			self.kernel = np.asarray([[-1,-1,-1], 
									  [0, 0, 0], 
									  [1, 1, 1]])
		elif kernel == 'identity':					# This is the implementation of identity kernel (will not change the original image at all).
			self.kernel = np.asarray([[0, 0, 0], 
									  [0, 1, 0], 
									  [0, 0, 0]])

		elif kernel == 'random':					# This will create a 3x3 kernel where the values are generated randomly (minimum value=-5, maximum value=5).
			np.random.seed(random_state)
			self.kernel = np.random.random_integers(low=-5, high=5, size=(3,3))

		# All attributes that we are going to create later.
		self.original = None
		self.resized = None
		self.zero_padded = None
		self.convolved = None

	# DESCRIPTION: This function creates zero padding to the image to be convolved so that the dimension of the resulting image will remain the same.
	# INPUT: Image to be zero-padded.
	# OUTPUT: An image with zero padding.
	def create_zero_padding(self, image):
		
		h_padding = np.zeros(image.shape[1])						# Horizontal padding (width).
		h_padded   = np.vstack((h_padding, image, h_padding))		# Create zero padding at the top and bottom of the original image.
		
		v_padding = np.zeros(image.shape[0]+2)						# Vertical padding (height). The shape is increased by 2 pixels since we have added two rows (at the top and bottom).
		all_padded = np.hstack((v_padding[:,np.newaxis], h_padded, v_padding[:,np.newaxis]))		# Create zero padding at the left and right of the original image.
		
		return all_padded       # The image that has had zero padding in all sides.

	# DESCRIPTION: Function to remove zero padding (this will be used to convert back the convolved image to the original size).
	# INPUT: image=An image that the zero padding will be removed.
	# OUTPUT: An image without zero-padding.
	def remove_padding(self, image):
		return image[1:-1,1:-1]			# Just a standard array slicing to remove zeros at all sides.

	# DESCRIPTION: Function to perform filtering on a single 3x3 region.
	# INPUT: sliced=A single 3x3 region.
	# OUTPUT: A new value for the current region center.
	def filtering(self, sliced):
		return np.dot(sliced.flatten(), self.kernel.flatten())

	# DESCRIPTION: Function to load and resize an image.
	# INPUTS: image_path=The directory of the picture to load. 
	# resolution=A value used to rescale the image. Best practice: use value less than 1 to make the image smaller since 3x3 kernel works best when the convolved image is small.
	def load_image(self, image_path, resolution=0.5):
		self.original = cv2.imread(image_path, flags=0)			# Function to load the image. The flags=0 is used to convert the image to grayscale.
		self.resized = cv2.resize(self.original, (int(self.original.shape[1]*resolution), int(self.original.shape[0]*resolution)))		# Resizing the image according to the resolution value.

		self.zero_padded = self.create_zero_padding(self.resized)		# Add zero padding to the resized image.
	
	# DESCRIPTION: Used to display the original image in grayscale.
	# INPUTS: No input.
	def show_original(self):
		plt.grid()
		plt.imshow(self.original, cmap='gray')
		plt.show()

	# DESCRIPTION: Used to display the resized image (notice the xticks and yticks values).
	# INPUTS: No input.
	def show_resized(self):
		plt.grid()
		plt.imshow(self.resized, cmap='gray')
		plt.show()

	# DESCRIPTION: Used to display the resized image with zero padding. The output image will look very similar to the output of show_resized() function
	# since the zero padding only gives 1 additional row/column on each side.
	# INPUT: No input.
	def show_zero_padded(self):
		plt.grid()
		plt.imshow(self.zero_padded, cmap='gray')
		plt.show()
	
	# DESCRIPTION: Function to display the convolved image (the final result).
	# INPUT: No input.
	def show_convolved(self):
		plt.grid()
		plt.imshow(self.convolved, cmap='gray')
		plt.show()

	# DESCRIPTION: Function to perform the entire convolution process.
	# INPUT: No input.
	def convolve(self):
		filtered_image = np.zeros(shape=(self.resized.shape[0], self.resized.shape[1]))
		for i in tqdm(range(1, self.resized.shape[0]-1)):		# For each row. Represents i-th row.
			for j in range(1, self.resized.shape[1]-1):			# For each column. Represents j-th column. This nested loop will make the kernel strides to the right first before going to the next row.
				sliced = self.resized[i-1:i+2, j-1:j+2]			# Take only the 3x3 image region.
				filtered_image[i,j] = self.filtering(sliced)	# Perform filtering on the current 3x3 region.
		
		self.convolved = self.remove_padding(filtered_image)    # Remove zero padding on the convolved image.

# DESCRIPTION: The class below is the implementation of Otsu thresholding.
# INPUT: Refer to the __init__() function below.
class Otsu:

	# DESCRIPTION: This function will be called every time a new Otsu object is initialized.
	# INPUT: No input.
	def __init__(self):
		# Below are the attributes that we are going to create later.
		self.image = None
		self.original_dimension = None
		self.flattened_image = None
		self.histogram = None
		self.pixel_id = None
		self.pixel_values_sum = None
		self.optimum_threshold = None
		self.thresholded_image = None
	
	# DESCRIPTION: Use this function to load an image from a specified path.
	# The loaded image will directly be converted to grayscale.
	# INPUT: image_path=The path of an image to load.
	def load_image(self, image_path):
		self.image = cv2.imread(image_path, flags=0)								# Load image as grayscale.
		self.original_dimension = (self.image.shape[0], self.image.shape[1])		# Save the original dimension since we are going to work with the flattened form starting from now on.

	# DESCRIPTION: After loading the image, we need to create the histogram prior to calculating the right threshold.
	# Note that the histogram is stored in "self.histogram". The histogram itself is in form of a single dimensional array that has the length of 256 representing each pixel intensity level.
	def create_histogram(self):
		self.flattened_image = self.image.flatten()						# Flatten the image (converting from 2-dimensional array to 1 dimension).
		self.histogram = self.count_unique(self.flattened_image)		# Using count_unique() function (created below) to create the histogram.
		self.pixel_id = np.arange(len(self.histogram))					# This is essentially just an array that has the value of 0, 1, 2, ..... 255.
		self.pixel_values_sum = np.sum(self.histogram)					# This is equivalent to the number of pixels in the image.
		
	# DESCRIPTION: Function to count the number of unique value in an array. In this case we assume that there will be 256 unique values (0 to 255).
	# INPUT: flattened_image=Grayscale image that we have converted to 1-dimensional array.
	# OUTPUT: 1-dimensional array with the length of 256 that represents pixel intensity level. The element of the array represents the number of pixels that have the corresponding intensity level.
	def count_unique(self, flattened_image):
		unique_values = [0] * 256						# Allocating list with the length of 256.
		for i in tqdm(range(len(flattened_image))):		# Iterate through all pixels in the array.
			value = flattened_image[i]
			unique_values[value] += 1					# We are going to increment the element of "unique_values" that corresponds to the current pixel intensity level.
			
		return unique_values

	# DESCRIPTION: Use this function in case we only want to find the optimum threshold only based on a histogram (without the actual image).
	# INPUT: histogram=A 1-dimensional array with a particular length where the index represents intensity level (from 0 to n).
	def load_histogram(self, histogram):
		self.histogram = histogram							# Since here we already have the histogram, then we don't need to use "count_unique()" function.
		self.pixel_id = np.arange(len(self.histogram))		# This is essentially just an array that has the value of 0, 1, 2, ..... 255.
		self.pixel_values_sum = np.sum(self.histogram)		# Equivalent to the number of pixels in the image (we do this instead of using "len(self.flattened_image)" because we in this case don't actually know the actual image).

	# DESCRIPTION: Function to display histogram.
	# INPUT: No input. We access the histogram through "self.histogram" attribute.
	def show_histogram(self):
		plt.title('Histogram')								# Defining figure title.
		plt.bar([i for i in range(256)], self.histogram)	# Displaying the graph.
		plt.show()
	
	### BACKGROUND PROCESSING FUNCTIONS (DARK PIXELS).
	# DESCRIPTION: Function to calculate weight of the background (please refer to the theory of Otsu).
	# INPUT: threshold=Threshold value.
	def calculate_background_weight(self, threshold):
		background_weight = np.sum(self.histogram[:threshold])/self.pixel_values_sum
		return background_weight

	# DESCRIPTION: Function to calculate mean of the background (please refer to the theory of Otsu).
	# INPUT: threshold=Threshold value.
	def calculate_background_mean(self, threshold):
		background_mean = np.dot(self.pixel_id[:threshold], self.histogram[:threshold]) / np.sum(self.histogram[:threshold])
		return background_mean

	# DESCRIPTION: Function to calculate variance of the background (please refer to the theory of Otsu).
	# INPUT: threshold=Threshold value. background_mean=The output of "calculate_background_mean()" function.
	def calculate_background_variance(self, threshold, background_mean):
		sum = 0
		for px_no, px_val in zip(self.pixel_id[:threshold], self.histogram[:threshold]):
			sum += ((px_no-background_mean)**2) * px_val

		background_variance = sum/np.sum(self.histogram[:threshold])
		return background_variance

	### FOREGROUND PROCESSING FUNCTIONS (BRIGHT PIXELS).
	# DESCRIPTION: Function to calculate weight of the foreground (please refer to the theory of Otsu).
	# INPUT: threshold=Threshold value.
	def calculate_foreground_weight(self, threshold):
		foreground_weight = np.sum(self.histogram[threshold:])/self.pixel_values_sum
		return foreground_weight

	# DESCRIPTION: Function to calculate mean of the foreground (please refer to the theory of Otsu).
	# INPUT: threshold=Threshold value.
	def calculate_foreground_mean(self, threshold):
		foreground_mean = np.dot(self.pixel_id[threshold:], self.histogram[threshold:]) / np.sum(self.histogram[threshold:])
		return foreground_mean

	# DESCRIPTION: Function to calculate variance of the foreground (please refer to the theory of Otsu).
	# INPUT: threshold=Threshold value. foreground=The output of "calculate_foreground_mean()" function.
	def calculate_foreground_variance(self, threshold, foreground_mean):
		sum = 0
		for px_no, px_val in zip(self.pixel_id[threshold:], self.histogram[threshold:]):
			sum += ((px_no-foreground_mean)**2) * px_val

		foreground_variance = sum/np.sum(self.histogram[threshold:])
		return foreground_variance

	# DESCRIPTION: Function to calculate WCV (Within Class Variance) given a threshold value.
	# INPUT: threshold=Threshold value.
	def calculate_wcv(self, threshold):
		background_weight = self.calculate_background_weight(threshold)
		background_mean = self.calculate_background_mean(threshold)
		background_variance = self.calculate_background_variance(threshold, background_mean)

		foreground_weight = self.calculate_foreground_weight(threshold)
		foreground_mean = self.calculate_foreground_mean(threshold)
		foreground_variance = self.calculate_foreground_variance(threshold, foreground_mean)

		wcv = (background_weight * background_variance) + (foreground_weight * foreground_variance)		# This is just the implementation of WCV formula. Please refer to the theory of Otsu.
		return wcv
	
	# DESCRIPTION: Function to find the optimum threshold for the image or histogram.
	# Note that a warning may appear on your screen since there is a division-by-zero operation. 
	# But we can just ignore the problem since it only occurs when the program is at threshold=0.
	# INPUT: No input.
	def find_optimum_threshold(self):
		wcvs = []										# Allocating an empty list to store the WCV of each threshold value. Later this will have the length of 256.
		for i in tqdm(range(len(self.histogram))):		# Iterate through all all intensity levels (this will be iterating 256 times).
			wcv = self.calculate_wcv(i)					# Calculate WCV using "calculate_wcv()" function. Current index becomes the threshold value.
			wcvs.append(wcv)							# Put the result to "wcvs" list that we allocated earlier.

		wcvs = np.asarray(wcvs)							# Converting "wcvs" list to Numpy array. This step is not very necessary though.
		argsorted_wcvs = np.argsort(wcvs)				# Sort the "wcvs", smallest value will be at the leftmost, but we will only take the index number.

		self.optimum_threshold = argsorted_wcvs[0]		# The optimum threshold is the intensity level that has the lowest WCV.

		return argsorted_wcvs[0], wcvs[argsorted_wcvs[0]]		# Akan keluar warning karena ada nan value nya. Tapi bisa diabaikan saja.
	
	# DESCRIPTION: Function to display the original image. You will not be able to run this method if you don't provide an image at the first place.
	# INPUT: No input.
	def show_original_image(self):
		plt.title('Original image')
		plt.xticks([])
		plt.yticks([])
		plt.imshow(self.image, cmap='gray')
		plt.show()

	# DESCRIPTION: Function to display the binary image, where the threshold is determined automatically using Otsu method that we just implemented.
	# INPUT: No input.
	def show_thresholded_image(self):
		thresholded_image_flat = np.zeros(len(self.flattened_image))		# Allocating an all-zero 1-dimensional array. This is equivalent to allocating an empty list.
		for i in tqdm(range(len(self.flattened_image))):					# Iterate through all pixels in the flattened image.
			if self.flattened_image[i] >= self.optimum_threshold:			# If the current intensity value is greater than or equals to the threshold, then convert it to a white pixel (255).
				thresholded_image_flat[i] = 255
			else:															# If the current intensity is less than the threshold, then convert to black (0).
				thresholded_image_flat[i] = 0

		self.thresholded_image = thresholded_image_flat.reshape(self.original_dimension)		# Convert to the original image shape (converting back to 2-dimensional array).
		self.thresholded_image = self.thresholded_image // 255									# Normalize the value such that the resulting binary image consists of 0 and 1 values (not 0 and 255).

		plt.title('Thresholded image')
		plt.xticks([])
		plt.yticks([])
		plt.imshow(self.thresholded_image, cmap='gray')
		plt.show()

# DESCRIPTION: The class below is the implementation of how HOG features are extracted from an image.
# It is important to know that currently it only compatible with an image that has the dimension of 128 x 64.
# Any image of different size will automatically be resized to 128 x 64.
# INPUT: Refer to the __init__() function below.
class HOG:

	# DESCRIPTION: This function will got called once a HOG class is instantiated.
	# INPUT: No input.
	def __init__(self):
		# Below are all attributes of HOG class.
		self.original_image = None				# The original image (already resized to 128 x 64).
		self.horizontal_derivative = None		# The resulting image after applying horizontal_edge_kernel.
		self.vertical_derivative = None			# The resulting image after applying vertical_edge_kernel.
		self.edge_magnitude = None				# abs(horizontal derivative) + abs(vertical derivative).
		self.edge_direction = None				# atan(v/h) * (180/pi).
		self.cells_histograms = None			# Histogram of all cells.
		self.blocks_histograms = None			# Histogram of all blocks. This is our HOG features.
		self.cell_size = 8						# We will divide the image into cells of 8 x 8 pixels.
		self.block_size = 2						# We will divide the image into blocks of 2 x 2 cells

		# Kernel for taking horizontal derivative.
		self.horizontal_edge_kernel = np.asarray([[0, 0, 0], 
												  [-1,0, 1], 
												  [0, 0, 0]])

		# Kernel for taking vertical derivative.
		self.vertical_edge_kernel = np.asarray([[0,-1, 0], 
												[0, 0, 0], 
												[0, 1, 0]])
	
	# DESCRIPTION: Function to load image and directly convert it to grayscale and resize to 128 x 64.
	# INPUT: image_path=The directory location of the image.
	def load_image(self, image_path):
		image = cv2.imread(image_path, flags=0)			# Read the image as grayscale.
		image = cv2.resize(image, dsize=(64, 128))		# Resize to 128 x 64 (h x w). Note that cv2 by default takes the input of w x h instead.
		self.original_image = image						# Store the processed image to the original_image attribute.
	
	# DESCRIPTION: Function to calculate horizontal derivative.
	# INPUT: A 3 x 3 region of the currently selected (sliced) image.
	def first_derivative_h(self, sliced):
		return np.dot(sliced.flatten(), self.horizontal_edge_kernel.flatten())


	# DESCRIPTION: Function to calculate vertical derivative.
	# INPUT: A 3 x 3 region of the currently selected (sliced) image.
	def first_derivative_v(self, sliced):
		return np.dot(sliced.flatten(), self.vertical_edge_kernel.flatten())

	# DESCRIPTION: Function to start convolution process (calculating derivatives, edge magnitude and edge direction).
	# Keep in mind that even though "convolve()" method here is recycled from "Convolution" class, yet there are several differences here and there.
	# One of the difference is that here zero padding is not applied.
	# INPUT: No input.
	def convolve(self):

		# Initially both horizontal and vertical derivative images are all zeros. 
		# Those zero values are then going to get updated using "first_derivative_v()" and "first_derivative_h()" function.
		self.horizontal_derivative = np.zeros(shape=(self.original_image.shape[0], self.original_image.shape[1]))
		self.vertical_derivative = np.zeros(shape=(self.original_image.shape[0], self.original_image.shape[1]))

		for i in tqdm(range(1, self.original_image.shape[0]-1)):					# Iterate downwards.
			for j in range(1, self.original_image.shape[1]-1):						# Iterate to the right. This nested loop will stride to the right first before moving to the next row.
				sliced = self.original_image[i-1:i+2, j-1:j+2]						# Take only the 3x3 image region.
				self.horizontal_derivative[i,j] = self.first_derivative_h(sliced)	# Perform filtering on the current 3x3 region using horizontal edge kernel.
				self.vertical_derivative[i,j] = self.first_derivative_v(sliced)		# Perform filtering on the current 3x3 region usin vertical edge kernel.

		self.edge_magnitude = np.abs(self.horizontal_derivative) + np.abs(self.vertical_derivative)		# Combining horizontal and vertical derivative.
		edge_direction = np.arctan(self.vertical_derivative/self.horizontal_derivative) * (180/np.pi)	# Finding the gradient direction.
		self.edge_direction = np.nan_to_num(edge_direction) + 90										# The resulting gradient direction is added by 90 since originally it ranges between -90 to 90 instead of 0 to 180.

	# DESCRIPTION: Function to determine the right bin given an angle.
	# INPUT: angle=Angle in degree ranging from 0 to 180 (inclusive).
	# OUTPUT: A bin code (determines the index of an array).
	def determine_bin(self, angle):
		if 0 <= angle and angle < 20:
			return 0
		elif 20 <= angle and angle < 40:
			return 1
		elif 40 <= angle and angle < 60:
			return 2
		elif 60 <= angle and angle < 80:
			return 3
		elif 80 <= angle and angle < 100:
			return 4
		elif 100 <= angle and angle < 120:
			return 5
		elif 120 <= angle and angle < 140:
			return 6
		elif 140 <= angle and angle < 160:
			return 7
		elif 160 <= angle and angle <= 180:
			return 8

	# DESCRIPTION: Function to create histogram. This will return a 9-element array containing the number of occurences of a particular angle range.
	# So far the "magnitude_region" parameter is still useless since in this case we don't take into account edge magnitude to actually construct the features.
	# As you can see the "determine_bin()" function above, the bin is still determined only by the angle.
	# INPUT: magnitude_region=A cell of 8x8 pixels taken from the image showing edge magnitude only.
	# INPUT: direction_region=A cell of 8x8 pixels taken from the image showing edge direction only.
	# OUTPUT: A histogram of 9 bins.
	def create_histogram(self, magnitude_region, direction_region):
		histogram = np.zeros(9)								# Allocating all-zero array.
		flattened_magnitude = magnitude_region.flatten()
		flattened_direction = direction_region.flatten()

		for i in range(len(flattened_magnitude)):
			bin = self.determine_bin(flattened_direction[i])		# Find the right bin for the current angle.
			histogram[bin] += 1										# Increase the bin value of the selected index.
		
		return histogram				# Return the resulting histogram.
	
	# DESCRIPTION: This function is used to create a histogram for each cell. Each cell consists of 8x8 pixels. Thus the sum of the histogram of every cell will be 64.
	# INPUT: No input.
	def create_histogram_for_every_cell(self):	
		cells_histograms = []			# Allocating empty list for storing histograms of each cell.
		for i in range(0, self.edge_magnitude.shape[0], self.cell_size):						# Stride downwards, jumps every 8 pixels (according to "cell_size").
			for j in range(0, self.edge_magnitude.shape[1], self.cell_size):					# Stride to the right, jumps every 8 pixels (according to "cell_size").
				cell_magnitude = self.edge_magnitude[i:i+self.cell_size, j:j+self.cell_size]	# Take 8x8 region from the image containing edge magnitude.
				cell_direction = self.edge_direction[i:i+self.cell_size, j:j+self.cell_size]	# Take 8x8 region from the image containing edge direction.
				cell_histogram = self.create_histogram(cell_magnitude, cell_direction)			# Create the histogram for every cell. If the cell size is 8x8, then the sum of the histogram array will be 64.
				cells_histograms.append(cell_histogram)		# Every element of this list is a histogram, so at the end of the day this will be a 2-dimensional list.

		cells_histograms = np.asarray(cells_histograms)			# Convert the list to Numpy array.
		cells_histograms = cells_histograms.reshape(16, 8, 9)	# Our original image has the size of 128x64, thus if it is grouped into 8x8 cells, then we will got 16 cells x 8 cells, where each of those cells consists of a histogram of 9 elements.
		self.cells_histograms = cells_histograms				# Store the histogram of all cells to "cells_histograms" attribute.

	# DESCRIPTION: Each block consists of 2x2 cells that strides to the right first before going to the next row.
	# Note that the stride is 1, hence these blocks are overlapping. The default "block_size" is 2x2, thus a single block consists of 4 cells.
	# INPUT: No input.
	def create_histogram_for_every_block(self):
		blocks_histograms = []			# Allocating emtpy list for storing histograms of each block.
		for i in range(self.cells_histograms.shape[0]-1):				# Stride downwards.
			for j in range(self.cells_histograms.shape[1]-1):			# Stride to the right. The nested loop goes to the right first prior to moving to the next row.
				block_histogram = self.cells_histograms[i:i+self.block_size, j:j+self.block_size]		# The topleft cell becomes the center.
				blocks_histograms = np.append(blocks_histograms, block_histogram)						# Put all histograms to "blocks_histogram" array.
		
		self.blocks_histograms = blocks_histograms														# Then store it in the "blocks_histograms" attribute.
																										# Note that we don't convert to the original dimension like what I did in the "create_histogram_for_every_cell()" function since it is not really necessary to do so.
																										# The "blocks_histograms" attribute is the HOG feature.

	# DESCRIPTION: Use this function in case you don't want to call the methods one by one in order to take the HOG features.
	# INPUT: No input.
	# OUTPUT: HOG features.
	def extract_features(self):
		self.convolve()								# Start the convolution process. This will produce "horizontal_derivative", "vertical_derivative", "edge_magnitude", and "edge_direction" attribute.
		self.create_histogram_for_every_cell()		# Start creating the histogram of every cell. This produces "cells_histograms" attribute.
		self.create_histogram_for_every_block()		# Start concatenating the histogram of every 2x2 cells, where each cell consists of 9 bins. This produces "blocks_histograms" attribute.

		return self.blocks_histograms				# The "blocks_histograms" attribute contains the HOG features.

	# DESCRIPTION: Show the original image (already resized to 128x64 and grayscaled).
	# INPUT: No input.
	def show_original(self):
		plt.grid()
		plt.imshow(self.original_image, cmap='gray')
		plt.show()
	
	# DESCRIPTION: Show the horizontal derivative image.
	# INPUT: No input.
	def show_horizontal_derivative(self):
		plt.grid()
		plt.imshow(self.horizontal_derivative, cmap='gray')
		plt.show()

	# DESCRIPTION: Show the vertical derivative image.
	# INPUT: No input.
	def show_vertical_derivative(self):
		plt.grid()
		plt.imshow(self.vertical_derivative, cmap='gray')
		plt.show()
	
	# DESCRIPTION: Show the overall edge magnitude (abs(vertical_d) + abs(horizontal_d)).
	# INPUT: No input.
	def show_edge_magnitude(self):
		plt.grid()
		plt.imshow(self.edge_magnitude, cmap='gray')
		plt.show()
	
	# DESCRIPTION: Show the edge direction. Note that in this version the edge direction does not take into account the amount of edge magnitude.
	# INPUT: No input.
	def show_edge_direction(self):
		plt.grid()
		plt.imshow(self.edge_direction, cmap='gray')
		plt.show()
	