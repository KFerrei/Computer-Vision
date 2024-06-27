### Supporting code for Computer Vision Assignment 1
### See "Assignment 1.ipynb" for instructions

import math

import numpy as np
from skimage import io
import matplotlib.pyplot as plt

def load(img_path):
    """Loads an image from a file path.

    HINT: Look up `skimage.io.imread()` function.
    HINT: Converting all pixel values to a range between 0.0 and 1.0
    (i.e. divide by 255) will make your life easier later on!

    Inputs:
        image_path: file path to the image.

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    
    out = None
    out = io.imread(img_path)/255
    return out

def print_stats(image):
    """ Prints the height, width and number of channels in an image.
        
    Inputs:
        image: numpy array of shape(image_height, image_width, n_channels).
        
    Returns: none
                
    """
    shape = image.shape
    height, width = shape[:2]
    channel = 1
    if len(shape) > 2:
        channel = shape[2]
    print("Height:", height )
    print("Width:", width )
    print("Channel:", channel )
    
    return None


def crop(image, start_row, start_col, num_rows, num_cols):
    """Crop an image based on the specified bounds. Use array slicing.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        start_row (int): The starting row index 
        start_col (int): The starting column index 
        num_rows (int): Number of rows in our cropped image.
        num_cols (int): Number of columns in our cropped image.

    Returns:
        out: numpy array of shape(num_rows, num_cols, 3).
    """
    out = None
    size = image.shape
    if (start_row+num_rows)>size[0]:
        print("Warning: the number of rows desired of the cropped image is greater than that of the original image")
    if (start_col+num_cols)>size[1]:
        print("Warning: the number of columns desired of the cropped image is greater than that of the original image")
    out = image[start_row:start_row+num_rows, start_col:start_col+num_cols]

    return out


def change_contrast(image, factor):
    """Change the value of every pixel by following

                        x_n = factor * (x_p - 0.5) + 0.5

    where x_n is the new value and x_p is the original value.
    Assumes pixel values between 0.0 and 1.0 
    If you are using values 0-255, change 0.5 to 128.

    Inputs:
        image: numpy array of shape(image_height, image_width, 3).
        factor (float): contrast adjustment

    Returns:
        out: numpy array of shape(image_height, image_width, 3).
    """
    out = None
    out = np.clip(factor * (image - 0.5) + 0.5, 0, 1)
    return out


def resize(input_image, output_rows, output_cols):
    """Resize an image using the nearest neighbor method.
    i.e. for each output pixel, use the value of the nearest input pixel after scaling

    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.
        output_rows (int): Number of rows in our desired output image.
        output_cols (int): Number of columns in our desired output image.

    Returns:
        np.ndarray: Resized image, with shape `(output_rows, output_cols, 3)`.
    """
    out = None
    input_rows, input_cols = input_image.shape[:2]
    if len(input_image.shape) == 2:
        channel = 1
    else:
        channel = input_image.shape[2]
    out = np.zeros([output_rows, output_cols, channel])
    for i in range(output_rows):
        for j in range(output_cols):
            out[i,j] = input_image[int(i*input_rows/output_rows),
                                   int(j*input_cols/output_cols)]
    return out

def greyscale(input_image):
    """Convert a RGB image to greyscale. 
    A simple method is to take the average of R, G, B at each pixel.
    Or you can look up more sophisticated methods online.
    
    Inputs:
        input_image: RGB image stored as an array, with shape
            `(input_rows, input_cols, 3)`.

    Returns:
        np.ndarray: Greyscale image, with shape `(output_rows, output_cols)`.
    """
    out = None
    # Luma coding
    out = 0.299 * input_image[:, :, 0] + 0.587 * input_image[:, :, 1] + 0.114 * input_image[:, :, 2]
    return out

def binary(grey_img, th):
    """Convert a greyscale image to a binary mask with threshold.
  
                  x_n = 0, if x_p < th
                  x_n = 1, if x_p > th
    
    Inputs:
        input_image: Greyscale image stored as an array, with shape
            `(image_height, image_width)`.
        th (float): The threshold used for binarization, and the value range is 0 to 1
    Returns:
        np.ndarray: Binary mask, with shape `(image_height, image_width)`.
    """
    out = None
    out = grey_img.copy()
    out[out > th] = 1
    out[out < th] = 0
    return out


def conv2D(image, kernel):
    """ Convolution of a 2D image with a 2D kernel. 
    Convolution is applied to each pixel in the image.
    Assume values outside image bounds are 0.
    
    Args:
        image: numpy array of shape (Hi, Wi).
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi).
    """
    out = None
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape
    padded_image = np.pad(image, ((kernel_height//2, kernel_height//2), (kernel_width//2, kernel_width//2)), mode='constant')
    out = np.zeros((image_height, image_width))
    kernel = np.flipud(np.fliplr(kernel)) 
    for y in range(image_height):
        for x in range(image_width):
            # Get the region of interest
            patch = padded_image[y:y+kernel_height, x:x+kernel_width]
            # Convolve the kernel with the region of interest
            out[y, x] = (kernel * patch).sum()
    return out

def test_conv2D():
    """ A simple test for your 2D convolution function.
        You can modify it as you like to debug your function.
    
    Returns:
        None
    """

    # Test code written by 
    # Simple convolution kernel.
    kernel = np.array(
    [
        [1,0,1],
        [0,0,0],
        [1,0,0]
    ])

    # Create a test image: a white square in the middle
    test_img = np.zeros((9, 9))
    test_img[3:6, 3:6] = 1

    # Run your conv_nested function on the test image
    test_output = conv2D(test_img, kernel)
    # Build the expected output
    expected_output = np.zeros((9, 9))
    expected_output[2:7, 2:7] = 1
    expected_output[5:, 5:] = 0
    expected_output[4, 2:5] = 2
    expected_output[2:5, 4] = 2
    expected_output[4, 4] = 3

    print(test_output)
    print(expected_output)


    # Test if the output matches expected output
    assert np.max(test_output - expected_output) < 1e-10, "Your solution is not correct."


def conv(image, kernel):
    """Convolution of a RGB or grayscale image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = None
    h,w = image.shape[:2]
    if len(image.shape) == 2:
        out = conv2D(image, kernel)
    else:
        out = np.copy(image)
        out[:,:,0] = conv2D(image[:,:,0], kernel)
        out[:,:,1] = conv2D(image[:,:,1], kernel)
        out[:,:,2] = conv2D(image[:,:,2], kernel)
        out = np.clip(out,0,1)
    return out

    
def gauss2D(size, sigma):

    """Function to mimic the 'fspecial' gaussian MATLAB function.
       You should not need to edit it.
       
    Args:
        size: filter height and width
        sigma: std deviation of Gaussian
        
    Returns:
        numpy array of shape (size, size) representing Gaussian filter
    """

    x, y = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    g = np.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g/g.sum()


def corr(image, kernel):
    """Cross correlation of a RGB image with a 2D kernel
    
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        kernel: numpy array of shape (Hk, Wk). Dimensions will be odd.

    Returns:
        out: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    out = None
    if len(kernel.shape) == 2:
            kernel = np.flip(kernel)
            out = conv(image, kernel)
    else:
        out = conv2D(image[:,:,0], np.flip(kernel[:,:,0])) + conv2D(image[:,:,1], np.flip(kernel[:,:,1]))+conv2D(image[:,:,2], np.flip(kernel[:,:,2]))
    return out

def simple_point_processing(image):
    """
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
    """
    
    list_images = np.array([np.copy(image)]*8)
    list_names  = ["original", "darken", "lower_contrast", 
                   "non_linear_lower_contrast", "invert", "lighten",
                   "raise_contrast", "non_linear_raise_contrast"]
    list_images[1] = np.clip(list_images[1]-0.5,0,1)
    list_images[2] = np.clip(list_images[2]/2,0,1)
    list_images[3] = np.clip(list_images[3]**(1/3),0,1)
    list_images[4] = np.clip(1-list_images[4],0,1)
    list_images[5] = np.clip(list_images[5]+0.5,0,1)
    list_images[6] = np.clip(list_images[6]*2,0,1)
    list_images[7] = np.clip(list_images[7]**2,0,1)
    f, axarr = plt.subplots(2,4, figsize=(10, 4))
    k = 0
    for i in range(2):
        for j in range(4):
            axarr[i,j].imshow(list_images[k])
            axarr[i,j].set_title(list_names[k])
            axarr[i,j].axis('off')
            k += 1

def simple_point_processing_one_channel(image, channel):
    """
    Args:
        image: numpy array of shape (Hi, Wi, 3) or (Hi, Wi)
        channel : channel to process
    """
    list_images = np.array([np.copy(image)]*8)
    list_names  = ["original", "darken", "lower_contrast", 
                   "non_linear_lower_contrast", "invert", "lighten",
                   "raise_contrast", "non_linear_raise_contrast"]
    list_images[1][:,:,channel] = np.clip(list_images[1][:,:,channel]-0.5,0,1)
    list_images[2][:,:,channel] = np.clip(list_images[2][:,:,channel]/2,0,1)
    list_images[3][:,:,channel] = np.clip(list_images[3][:,:,channel]**(1/3),0,1)
    list_images[4][:,:,channel] = np.clip(1-list_images[4][:,:,channel],0,1)
    list_images[5][:,:,channel] = np.clip(list_images[5][:,:,channel]+0.5,0,1)
    list_images[6][:,:,channel] = np.clip(list_images[6][:,:,channel]*2,0,1)
    list_images[7][:,:,channel] = np.clip(list_images[7][:,:,channel]**2,0,1)
    f, axarr = plt.subplots(2,4,figsize=(10, 4))
    k = 0
    for i in range(2):
        for j in range(4):
            axarr[i,j].imshow(list_images[k])
            axarr[i,j].set_title(list_names[k])
            axarr[i,j].axis('off')
            k += 1