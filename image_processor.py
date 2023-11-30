from skimage.morphology import binary_erosion, binary_opening
from visualizer import plot_images_histograms, save_images
import cv2
from matplotlib import pyplot as plt
import numpy as np


image_paths = ["images/image1.JPG",
               "images/image2.JPG", "images/image3.JPG"]


def get_base_frame(images):
    return np.median(images, axis=0).astype(np.uint8)


def load_images(image_paths):
    return [cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB) for path in image_paths]


def detect_cars(images):
    imamges_grey = [cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                    for image in images]

    differences = [np.where(cv2.absdiff(image1, image2) > 64, 1, 0) for i, image1 in enumerate(
        imamges_grey) for j, image2 in enumerate(imamges_grey) if i < j]

    return [binary_opening(binary_erosion(binary_erosion(difference)))
            for difference in differences]


def enhance_differences(differences):
    return [np.where(cv2.dilate(difference.astype(np.uint8), np.ones(
        (69-32, 69-32), np.uint8), iterations=1) > 0, 1, 0) for difference in differences]


def get_connected_components(masks):
    return [cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 4) for mask in masks]


def remove_cars(images):
    base_frame = get_base_frame(images)
    differences = detect_cars(images)
    masks = enhance_differences(differences)
    connected_components = get_connected_components(masks)
    for i, connected_component in enumerate(connected_components):
        num_labels, labels, stats, centroids = connected_component
        for j, (stat, centroid) in enumerate(zip(stats[1:], centroids[1:])):
            left, top, width, height, area = stat
            cars = [image[top:top+height, left:left+width] for image in images]
            save_images(np.concatenate(cars, axis=1), f"cars/car_{i}_{j}")

            base_frame[top:top+height, left:left+width] = cars[np.argmin(
                [np.sum(image - np.full_like(image, fill_value=(120, 130, 130))) for image in cars], axis=0)]

    plot_images_histograms(images)
    save_images(base_frame, "base_frame")
    save_images(differences, "differences")
    save_images(masks, "masks")
    save_images(base_frame, "base_frame_without_cars")


if __name__ == "__main__":
    remove_cars(load_images(image_paths))
