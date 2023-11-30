from skimage.morphology import binary_erosion, binary_opening
import cv2
from matplotlib import pyplot as plt
import numpy as np


image_paths = ["images/image1.JPG",
               "images/image2.JPG", "images/image3.JPG"]


def plot_images_histograms(images):
    for i, image in enumerate(images):
        plt.hist(image.ravel(), bins=256, color='orange', )
        plt.hist(image[:, :, 0].ravel(), bins=256, color='red', alpha=0.5)
        plt.hist(image[:, :, 1].ravel(), bins=256, color='Green', alpha=0.5)
        plt.hist(image[:, :, 2].ravel(), bins=256, color='Blue', alpha=0.5)
        plt.xlabel('Intensity Value')
        plt.ylabel('Count')
        plt.legend(
            ['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
        plt.savefig(f"histogram_{i}.png")
        plt.clf()


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
    plot_images_histograms(images)

    base_frame = get_base_frame(images)
    plt.imsave("base_frame.JPG", base_frame)

    differences = detect_cars(images)
    for i, difference in enumerate(differences):
        plt.imsave(f"difference_{i}.JPG", difference)

    masks = enhance_differences(differences)
    for i, mask in enumerate(masks):
        plt.imsave(f"mask_{i}.JPG", mask)

    connected_components = get_connected_components(masks)
    for i, connected_component in enumerate(connected_components):
        num_labels, labels, stats, centroids = connected_component
        for j, (stat, centroid) in enumerate(zip(stats[1:], centroids[1:])):
            left, top, width, height, area = stat
            cars = [image[top:top+height, left:left+width] for image in images]
            cv2.imwrite(
                f"cars/car_{i}_{j}.JPG", cv2.cvtColor(np.concatenate(cars, axis=1), cv2.COLOR_BGR2RGB))

            base_frame[top:top+height, left:left+width] = cars[np.argmin(
                [np.sum(image - np.full_like(image, fill_value=(120, 130, 130))) for image in cars], axis=0)]

    plt.imsave("no_cars.JPG", base_frame)


if __name__ == "__main__":
    remove_cars(load_images(image_paths))
