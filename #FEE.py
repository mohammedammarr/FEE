import cv2
import numpy as np
import matplotlib.pyplot as plt




def increase_contrast(image, alpha=1.5, beta=0):
    adjusted_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted_image


def reduce_brightness(image, value=30):
    hls_image = cv2.cvtColor(image, cv2.COLOR_BGR2HLS)

    hls_image[:, :, 1] = np.clip(hls_image[:, :, 1] - value, 0, 255)

    brightness_reduced_image = cv2.cvtColor(hls_image, cv2.COLOR_HLS2BGR)

    return brightness_reduced_image


def plot_histograms(original_image, processed_image=None):
    original_hist_red = cv2.calcHist([original_image], [0], None, [256], [0, 256])
    original_hist_green = cv2.calcHist([original_image], [1], None, [256], [0, 256])
    original_hist_blue = cv2.calcHist([original_image], [2], None, [256], [0, 256])

    original_hist_red = original_hist_red.ravel()
    original_hist_green = original_hist_green.ravel()
    original_hist_blue = original_hist_blue.ravel()

    if processed_image is not None:
        processed_hist_red = cv2.calcHist([processed_image], [0], None, [256], [0, 256])
        processed_hist_green = cv2.calcHist([processed_image], [1], None, [256], [0, 256])
        processed_hist_blue = cv2.calcHist([processed_image], [2], None, [256], [0, 256])

        processed_hist_red = processed_hist_red.ravel()
        processed_hist_green = processed_hist_green.ravel()
        processed_hist_blue = processed_hist_blue.ravel()

    
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 2, 1)
    plt.plot(original_hist_red, color='red', label='Red')
    plt.plot(original_hist_green, color='green', label='Green')
    plt.plot(original_hist_blue, color='blue', label='Blue')
    plt.title('Original Image Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.legend()

    if processed_image is not None:
        plt.subplot(1, 2, 2)
        plt.plot(processed_hist_red, color='red', label='Red')
        plt.plot(processed_hist_green, color='green', label='Green')
        plt.plot(processed_hist_blue, color='blue', label='Blue')
        plt.title('Processed Image Histogram')
        plt.xlabel('Pixel Value')
        plt.ylabel('Frequency')
        plt.legend()

    plt.show()

image_path = "D:\lion.webp"
image = cv2.imread(image_path)


contrast_increased_image = increase_contrast(image)

brightness_reduced_image = reduce_brightness(contrast_increased_image)

cv2.imshow('Original Image', image)
cv2.imshow('Processed Image', brightness_reduced_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

plot_histograms(image, brightness_reduced_image)