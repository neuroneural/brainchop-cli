import numpy as np
import nibabel as nib
from utils._interpolation import zoom

def conform(img, output_shape=(256, 256, 256)):
    """
    Conform a NIfTI image by padding to a cube and then resizing to the specified shape.
    
    Parameters:
    - img: nibabel.nifti1.Nifti1Image, input NIfTI image
    - output_shape: tuple, desired output shape (default is (256, 256, 256))
    """
    # Load the image data
    data = img.get_fdata()
    
    # Find the maximum dimension
    max_dim = max(data.shape)
    
    # Create a cubic array filled with zeros
    cubic_data = np.zeros((max_dim, max_dim, max_dim))
    
    # Calculate padding
    pad_x = (max_dim - data.shape[0]) // 2
    pad_y = (max_dim - data.shape[1]) // 2
    pad_z = (max_dim - data.shape[2]) // 2
    
    # Place the original data in the center of the cubic array
    cubic_data[pad_x:pad_x+data.shape[0], 
               pad_y:pad_y+data.shape[1], 
               pad_z:pad_z+data.shape[2]] = data
    
    # Calculate zoom factors
    zoom_factors = np.array(output_shape) / np.array(cubic_data.shape)
    
    # Resize to the desired shape
    resized_data = zoom(cubic_data, zoom_factors, order=1)
    
    # Update the affine transformation
    new_affine = img.affine.copy()
    original_zooms = img.header.get_zooms()
    new_zooms = np.array(original_zooms) * (np.array(cubic_data.shape) / np.array(output_shape))
    new_affine[:3, :3] = np.diag(new_zooms)
    
    # Create new image
    new_img = nib.Nifti1Image(resized_data, new_affine)
    return new_img

if __name__ == "__main__":
    input_img = nib.load(input("Path to nifti file: "))
    # Apply the conform function
    output_img = conform(input_img)
    # Save the result
    nib.save(output_img, input("Path to output: "))
