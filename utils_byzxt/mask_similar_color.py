from PIL import Image, ImageEnhance
import numpy as np
from skimage import color

# Load image
image = Image.open('/mnt/sda/zxt/z_datas/imgs/3_style_data/fu_rong.jpg')

# Enhance the saturation
enhancer = ImageEnhance.Color(image)
enhanced_image = enhancer.enhance(2.0)

image_np = np.array(enhanced_image)

# Convert the image to LAB color space using skimage
lab_image = color.rgb2lab(image_np)

# Define target RGB color
target_rgb = (217,137,0)

# Convert target RGB color to LAB using skimage
target_lab = color.rgb2lab(np.array([[target_rgb]], dtype=np.uint8))[0][0]

# Define threshold for color similarity
color_threshold = 20

# Create mask for selected color blocks
mask = np.zeros(image_np.shape[:2], dtype=np.uint8)

# Iterate through each pixel
for i in range(image_np.shape[0]):
    for j in range(image_np.shape[1]):
        lab_pixel = lab_image[i, j]
        dist = np.linalg.norm(lab_pixel - target_lab)
        if dist < color_threshold:
            mask[i, j] = 255

# Apply mask to the original image
result_image_np = np.zeros_like(image_np)
result_image_np[mask == 255] = image_np[mask == 255]

# Convert result back to image
result_image = Image.fromarray(result_image_np)

# Display selected color blocks
result_image.show()
