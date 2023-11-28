import cv2
import numpy as np
from skimage import segmentation
import matplotlib.pyplot as plt


def remove_cars(image_paths):
    # Load the images
    images = [cv2.imread(image_path) for image_path in image_paths]

    # Convert images to RGB (OpenCV uses BGR by default)
    images_rgb = [cv2.cvtColor(image, cv2.COLOR_BGR2RGB) for image in images]

    # Combine the images into one numpy array
    combined_image = np.stack(images_rgb, axis=0)

    # Convert the images to a 2D array
    combined_image_2d = combined_image.reshape(-1, 3)

    # Use k-means clustering to segment the image into 2 clusters (background and cars)
    kmeans = segmentation.slic(
        combined_image, n_segments=2, compactness=10, sigma=1)

    # Find the cluster with the cars
    car_cluster = 1 if np.sum(kmeans == 0) > np.sum(kmeans == 1) else 0

    # Create a mask for the cars
    car_masks = [(kmeans[i] == car_cluster).reshape(image.shape[:2])
                 for i, image in enumerate(images)]

    # Apply the mask to each image to remove the cars
    road_images = [np.where(car_mask[:, :, np.newaxis], 255, image)
                   for car_mask, image in zip(car_masks, images)]

    # Calculate the average image
    average_image = np.mean(road_images, axis=0).astype(np.uint8)

    # Display the average image
    plt.figure()
    plt.imshow(average_image)
    plt.axis('off')
    plt.title("Average Road Image")
    plt.show()

    return road_images


# Example usage
image_paths = ["project_one/images/image1.jpg",
               "project_one/images/image2.jpg", "project_one/images/image3.jpg"]
result_images = remove_cars(image_paths)
