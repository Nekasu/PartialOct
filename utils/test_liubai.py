import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import greycomatrix, greycoprops

# 读取图像
image = cv2.imread('/mnt/sda/zxt/z_datas/imgs/3_style_data/waterry_flower_enhanced.jpg')
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 转换为HSV颜色空间
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# 定义黄色的HSV范围
lower_yellow = np.array([15, 40, 140])
upper_yellow = np.array([40, 255, 255])

# 使用颜色阈值分割黄色部分
mask_color = cv2.inRange(hsv, lower_yellow, upper_yellow)

# 计算纹理特征
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
glcm = greycomatrix(gray, [1], [0], 256, symmetric=True, normed=True)
texture = greycoprops(glcm, 'contrast')

# 正则化纹理特征
texture = (texture - np.min(texture)) / (np.max(texture) - np.min(texture))
texture = (texture * 255).astype(np.uint8)

# 阈值化纹理特征
ret, mask_texture = cv2.threshold(texture, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# 结合颜色和纹理掩码
mask_combined = cv2.bitwise_or(mask_color, mask_texture)

# 形态学操作，清理和增强分割结果
kernel = np.ones((5, 5), np.uint8)
mask_cleaned = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel)
mask_cleaned = cv2.morphologyEx(mask_cleaned, cv2.MORPH_OPEN, kernel)

# 提取留白部分
result = cv2.bitwise_and(image_rgb, image_rgb, mask=mask_cleaned)

# 显示结果
plt.figure(figsize=(20, 10))
plt.subplot(2, 3, 1)
plt.imshow(image_rgb)
plt.title('Original Image')
plt.subplot(2, 3, 2)
plt.imshow(mask_color, cmap='gray')
plt.title('Mask of Yellow Parts (Color)')
plt.subplot(2, 3, 3)
plt.imshow(mask_texture, cmap='gray')
plt.title('Mask of Texture Parts')
plt.subplot(2, 3, 4)
plt.imshow(mask_combined, cmap='gray')
plt.title('Combined Mask')
plt.subplot(2, 3, 5)
plt.imshow(mask_cleaned, cmap='gray')
plt.title('Cleaned Mask')
plt.subplot(2, 3, 6)
plt.imshow(result)
plt.title('Detected White Space')
plt.show()
