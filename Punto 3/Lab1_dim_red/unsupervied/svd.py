import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_svd(image_file, output_folder, sv_num_list, show_plots=True):
    """Applies SVD to an image and reconstructs it with varying numbers of singular values.

    Args:
        image_file (str): Path to the input image file.
        output_folder (str): Path to the output folder for saving reconstructed images.
        sv_num_list (list): List of numbers of singular values to use for reconstruction.
        show_plots (bool, optional): Whether to display the reconstructed images. Defaults to True.
    """

    # Load the image in grayscale
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Perform SVD and visualize progressively reconstructed images
    for sv_num in sv_num_list:
        reconstructed_img = svd_reconstruction(img, sv_num)

        # Save the reconstructed image
        output_path = os.path.join(output_folder, f'reconstructed_svd_{sv_num}.jpg')
        cv2.imwrite(output_path, reconstructed_img)

        # Display the reconstructed image if requested
        if show_plots:
            plt.imshow(reconstructed_img, cmap='gray')
            plt.title(f'Reconstructed Image with {sv_num} Singular Values')
            plt.show()

def svd_reconstruction(image, sv_num):
    """Reconstructs an image using a specified number of singular values.

    Args:
        image (numpy.ndarray): The input image.
        sv_num (int): The number of singular values to use for reconstruction.

    Returns:
        numpy.ndarray: The reconstructed image.
    """

    u, s, v = np.linalg.svd(image, full_matrices=False)
    s[sv_num:] = 0  # Keep only the first 'sv_num' singular values
    reconstructed_img = np.dot(u, np.dot(np.diag(s), v))
    return np.uint8(reconstructed_img)

# Example usage
image_file = '/content/Fotoperfil_b&n.jpg'  # Replace with your image path
output_folder = '/content/Result/'
sv_num_list = [10, 20, 50, 100]  # Adjust as needed

apply_svd(image_file, output_folder, sv_num_list)


