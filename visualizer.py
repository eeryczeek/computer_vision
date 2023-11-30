import matplotlib.pyplot as plt


def plot_images_histograms(images, path):
    for i, image in enumerate(images):
        plt.hist(image.ravel(), bins=256, color='orange', )
        plt.hist(image[:, :, 0].ravel(), bins=256, color='red', alpha=0.5)
        plt.hist(image[:, :, 1].ravel(), bins=256, color='Green', alpha=0.5)
        plt.hist(image[:, :, 2].ravel(), bins=256, color='Blue', alpha=0.5)
        plt.xlabel('Intensity Value')
        plt.ylabel('Count')
        plt.legend(
            ['Total', 'Red_Channel', 'Green_Channel', 'Blue_Channel'])
        plt.savefig(f"{path}_{i}.JPG")
        plt.clf()


def save_images(images, path):
    if type(images) != list:
        plt.imsave(f"{path}.JPG", images)
        return None
    for i, image in enumerate(images):
        plt.imsave(f"{path}_{i}.JPG", image)
