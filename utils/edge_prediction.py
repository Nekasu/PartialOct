import cv2
import numpy as np
from sklearn.cluster import KMeans
from skimage import filters, color
import matplotlib.pyplot as plt

def segment_image(image_path, num_clusters=3):
    """
    Segment an image into different regions using color clustering and texture analysis.

    Parameters:
    - image_path (str): The path to the input image.
    - num_clusters (int): The number of color clusters for K-means clustering.

    Returns:
    - segmented_image (numpy.ndarray): The segmented image.
    - mask (numpy.ndarray): The mask differentiating the subject and background.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"The image at {image_path} could not be found.")
    
    # Convert image to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Reshape image for clustering
    pixel_values = image_rgb.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=num_clusters, random_state=0)
    labels = kmeans.fit_predict(pixel_values)
    centers = kmeans.cluster_centers_
    centers = np.uint8(centers)
    
    # Reshape labels to match the image shape
    segmented_image = centers[labels.flatten()]
    segmented_image = segmented_image.reshape(image_rgb.shape)
    
    # Convert the segmented image to grayscale
    gray_segmented = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2GRAY)
    
    # Apply texture filter (e.g., Sobel filter for edge detection)
    texture = filters.sobel(gray_segmented)
    
    # Threshold the texture image to create a binary mask
    _, mask = cv2.threshold(texture, 0.1, 1.0, cv2.THRESH_BINARY)
    
    # Apply mask to the original image
    mask = mask.astype(np.uint8)
    masked_image = cv2.bitwise_and(image_rgb, image_rgb, mask=mask)
    
    return segmented_image, masked_image, mask

# Example usage:
image_path = '/mnt/sda/zxt/z_datas/imgs/3_style_data/fu_rong.jpg'  # Replace with your image path

segmented_image, masked_image, mask = segment_image(image_path, num_clusters=7)

# Display the results
plt.figure(figsize=(15, 5))
plt.subplot(131), plt.imshow(segmented_image)
plt.title('Segmented Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132), plt.imshow(masked_image)
plt.title('Masked Image'), plt.xticks([]), plt.yticks([])
plt.subplot(133), plt.imshow(mask, cmap='gray')
plt.title('Mask'), plt.xticks([]), plt.yticks([])
plt.show()
