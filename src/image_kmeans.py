# -*- coding: utf-8 -*-
"""
Compendium of functions used to clusterize pixels in images and generate the comparative side
by side on a new image saved in the outputs folder.
"""

from typing import NoReturn, Tuple

import os
import numpy as np
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans

#%%
def flatten_image(image: np.ndarray) -> Tuple[np.ndarray, Tuple]:
    """ Function to flatten the image from (N, M, 3) to (N*M, 3). """
    original_shape = image.shape
    reshaped_image = image.reshape(original_shape[0]*original_shape[1], original_shape[2])
    return reshaped_image, original_shape

def unflatten_image(flattened_image: np.ndarray, target_shape: Tuple) -> np.ndarray:
    """ Function to restore the original shape of the image. """
    restored_image = flattened_image.reshape(target_shape)
    return restored_image

#%%
class KmeansImageModel:
    def __init__(self, image_object: np.ndarray, n_clusters: int = 5):
        """ The kmeans is fitted over the input image. """
        self.n_clusters = n_clusters
        self.fit_kmeans_model(image_object)

    def fit_kmeans_model(self, image_object: np.ndarray) -> NoReturn:
        """ Function which trains the kmean model and obtains the principal colors. """
        # Image with shape (N*M, 3) ready for the kmeans
        flattened_image, _ = flatten_image(image_object)
        # Kmeans model training
        self.kmeans = KMeans(self.n_clusters, random_state=7).fit(flattened_image)
        self.principal_colors = self.kmeans.cluster_centers_.astype(int)

    def predict_kmeans_model(self, image: np.ndarray) -> np.ndarray:
        """ Function to apply the trained kmeans model over images. """
        flattened_image, original_shape = flatten_image(image)
        # Kmeans label prediction
        cluster_labels = self.kmeans.predict(flattened_image)
        # We generate a matrix with the cluster center of each label
        discrete_colors_image = np.zeros_like(flattened_image)
        for i in range(self.n_clusters):
            discrete_colors_image[cluster_labels == i, :] = self.principal_colors[i]
        output_img = unflatten_image(discrete_colors_image, original_shape)
        return output_img

#%%
def compose_final_image(original_image: np.ndarray,
                        kmeans_image: np.ndarray,
                        original_first: bool = False) -> np.ndarray:
    """ Function to join two images horizontally. """
    left_img = original_image if original_first else kmeans_image
    right_img = original_image if not original_first else kmeans_image
    composed_image = np.concatenate([left_img, right_img[:, ::-1, :]], axis=1)
    return composed_image

#%%
def clusterize_image(image_path: str,
                     output_path: str,
                     n_clusters: int = 16,
                     original_first: bool = False) -> NoReturn:
    """ Function which implements all the steps: model training, evaluation and saving results. """
    # Training and prediction
    training_image = plt.imread(image_path)
    kmeans_model = KmeansImageModel(training_image, n_clusters=n_clusters)
    training_image_result = kmeans_model.predict_kmeans_model(training_image)
    # Final image composition
    composed_image = compose_final_image(original_image=training_image,
                                         kmeans_image=training_image_result,
                                         original_first=original_first)
    # We save the image as {original_name}_n{n_clusters}.png
    new_name = os.path.basename(image_path).replace('.', f'_n{n_clusters}.')
    savename = os.path.join(output_path, new_name)
    # Final results plot and saving
    plt.figure(figsize=(19, 9))
    plt.imshow(composed_image)
    plt.axis('off')
    plt.tight_layout()
    if savename:
        plt.savefig(savename, bbox_inches='tight')
    plt.show()

#%%
if __name__ == '__main__':
    # Params
    IMAGE_PATH = './data/pexels-julius-silver-753626.jpg'
    OUTPUT_PATH = './outputs/'
    N_CLUSTERS = 16
    ORIGINAL_FIRST = True
    # Execution
    clusterize_image(image_path=IMAGE_PATH,
                     output_path=OUTPUT_PATH,
                     n_clusters=N_CLUSTERS,
                     original_first=ORIGINAL_FIRST)
