import cv2
from matplotlib import pyplot as plt
import numpy as np
from skimage.morphology import binary_erosion, binary_opening


image_paths = ["project_one/images/image1.JPG",
               "project_one/images/image2.JPG", "project_one/images/image3.JPG"]


def get_base_frame(images):
    return np.median(images, axis=0).astype(np.uint8)


def load_images(image_paths):
    """
    Loads images from a list of paths.

    Args:
        image_paths (list): A list of paths to images.

    Returns:
        list: A list of images.
    """
    return [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in image_paths]


def detect_cars(images):
    """
    Detects cars in a set of images and returns the number of cars detected in each image.

    Args:
        image_paths (list): A list of paths to images.

    Returns:
        list: A list of positions of cars in each image.
    """
    base_frame = get_base_frame(images)
    plt.imsave("baseframe.JPG", base_frame)

    imamges_grey = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    for image in images]

    differences = [np.where(cv2.absdiff(image1, image2) > 64, 1, 0) for i, image1 in enumerate(
        imamges_grey) for j, image2 in enumerate(imamges_grey) if i != j and i < j]

    differences = [binary_opening(binary_erosion(binary_erosion(difference)))
                   for difference in differences]

    masks = [np.where(cv2.dilate(difference.astype(np.uint8), np.ones(
        (69-32, 69-32), np.uint8), iterations=1) > 0, 1, 0) for difference in differences]

    connected_components = [cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 4) for mask in masks]

    for i, connected_component in enumerate(connected_components):
        num_labels, labels, stats, centroids = connected_component
        for j, (stat, centroid) in enumerate(zip(stats[1:], centroids[1:])):
            cars = [image[stat[1]:stat[1] +
                          stat[3], stat[0]:stat[0]+stat[2]] for k, image in enumerate(images)]
            cv2.imwrite(
                f"cars/car_{i}_{j}_.JPG", cv2.cvtColor(np.concatenate(cars, axis=1), cv2.COLOR_BGR2RGB))

            best_car = np.argmin(
                [np.sum(image - np.full_like(image, fill_value=(120, 130, 130))) for image in cars], axis=0)
            base_frame[stat[1]:stat[1] +
                       stat[3], stat[0]:stat[0]+stat[2]] = cars[best_car]
            print(f'car {i}_{j} stats: {stat}')

    base_frame = cv2.GaussianBlur(base_frame, (421, 421), 0)
    plt.imsave("no_cars.JPG", base_frame)


if __name__ == "__main__":
    detect_cars(load_images(image_paths))
