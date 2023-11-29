import cv2
import numpy as np
import matplotlib.pyplot as plt


from sklearn.cluster import KMeans


def remove_cars(image_paths):
    images = [cv2.imread(image_path) for image_path in image_paths]

    mean_image = np.median(images, axis=0).astype(np.uint8)
    mean_image_rgb = cv2.cvtColor(mean_image, cv2.COLOR_BGR2RGB)

    # Convert the mean image to the Lab color space
    mean_image_lab = cv2.cvtColor(mean_image_rgb, cv2.COLOR_RGB2Lab)

    # Reshape the image to a 2D array of pixels
    pixel_array = mean_image_lab.reshape((-1, 3))

    # Apply k-means clustering to group the pixels into clusters
    kmeans = KMeans(n_init='auto').fit(pixel_array)

    # Identify the cluster that corresponds to the cars
    car_cluster = np.argmin(kmeans.cluster_centers_.sum(axis=1))

    # Create a mask that selects only the pixels in the car cluster
    car_mask = (kmeans.labels_ == car_cluster).reshape(mean_image.shape[:2])

    # Apply the mask to the mean image to remove the cars
    result_image = mean_image_rgb.copy()
    result_image[car_mask] = [255, 255, 255]  # Replace car pixels with white

    plt.imsave("result.jpg", result_image)
    return None


image_paths = ["project_one/images/image1.jpg",
               "project_one/images/image2.jpg", "project_one/images/image3.jpg"]
result_images = remove_cars(image_paths)
