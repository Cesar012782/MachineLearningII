import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def apply_svd(image_file, output_folder, sv_num_list):
    # Load the image
    img = cv2.imread(image_file, cv2.IMREAD_GRAYSCALE)

    # Create an output folder if it doesn't exist
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Perform SVD and visualize progressively reconstructed images
    for sv_num in sv_num_list:
        reconstructed_img = svd_reconstruction(img, sv_num)

        # Save the reconstructed image
        output_path = os.path.join(output_folder, f'reconstructed_svd_{sv_num}.jpg')
        cv2.imwrite(output_path, reconstructed_img)

def svd_reconstruction(image, sv_num):
    u, s, v = np.linalg.svd(image, full_matrices=False)
    s[sv_num:] = 0  # Keep only the first 'sv_num' singular values
    reconstructed_img = np.dot(u, np.dot(np.diag(s), v))
    return np.uint8(reconstructed_img)

# Example usage
image_file = '/content/Fotoperfil_b&n.jpg'
output_folder = '/content/Result/'
sv_num_list = [10, 20, 50, 100]  # Adjust as needed

apply_svd(image_file, output_folder, sv_num_list)

