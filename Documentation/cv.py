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