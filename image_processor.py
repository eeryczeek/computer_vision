from skimage.morphology import binary_erosion, binary_opening
from visualizer import plot_images_histograms, save_images
import cv2
import numpy as np


image_paths = ["images/image1.JPG", "images/image2.JPG", "images/image3.JPG"]


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
        (37, 37), np.uint8), iterations=1) > 0, 1, 0) for difference in differences]


def get_connected_components(masks):
    return [cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), 4) for mask in masks]


def calculate_average_car_pixel_value(differences, images):
    car_pixels = [np.where(diff[..., None], img, 0)
                  for diff, img in zip(differences, images)]
    return np.mean([np.mean(car, axis=(0, 1)) for car in car_pixels], axis=0) * 255


def get_difficult_areas(differences):
    difficult_areas = np.sum(differences, axis=0) / len(differences)
    return difficult_areas


def remove_cars(images):
    base_frame = get_base_frame(images)
    base_frame_without_cars = base_frame.copy()

    differences = detect_cars(images)
    average_car_pixel_value = calculate_average_car_pixel_value(
        differences, images)

    masks = enhance_differences(differences)
    connected_components = get_connected_components(masks)

    for i, connected_component in enumerate(connected_components):
        _, _, stats, _ = connected_component
        for j, stat in enumerate(stats[1:]):
            left, top, width, height, area = stat
            cars = [image[top:top+height, left:left+width] for image in images]
            save_images(np.concatenate(cars, axis=1), f"cars/car_{i}_{j}")

            base_frame_without_cars[top:top+height, left:left+width] = cars[np.argmin(
                [np.sum(image - np.full_like(image, fill_value=average_car_pixel_value)) for image in cars], axis=0)]

    difficult_areas1 = enhance_differences([np.where(cv2.absdiff(cv2.cvtColor(
        base_frame, cv2.COLOR_RGB2GRAY), cv2.cvtColor(base_frame_without_cars, cv2.COLOR_RGB2GRAY)) > 64, 1, 0)])[0]
    difficult_areas2 = get_difficult_areas(differences)

    save_images(base_frame, "results/base_frame")
    plot_images_histograms(images, "results/histograms")
    save_images(differences, "results/differences")
    save_images(difficult_areas1, 'results/difficult_areas1')
    save_images(difficult_areas2, 'results/difficult_areas2')
    save_images(masks, "results/masks")
    save_images(base_frame_without_cars, "results/base_frame_without_cars")


if __name__ == "__main__":
    remove_cars(load_images(image_paths))
