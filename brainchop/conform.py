import numpy as np
import nibabel as nib
from scipy.ndimage import zoom

def conform(img, output_shape=(256, 256, 256)):
  """
  Conform a NIfTI image by padding to a cube, resizing to the specified shape,
  and ensuring 1mm isotropic voxels.
  
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
  resized_data = zoom(cubic_data, zoom_factors, order=1, mode="nearest")
  
  # Update the affine transformation for 1mm isotropic voxels
  new_affine = np.eye(4)
  new_affine[:3, :3] = np.diag([1, 1, 1])  # Set voxel size to 1mm isotropic
  
  # Adjust the origin to maintain the center of the image
  original_center = img.affine.dot(np.hstack((np.array(data.shape) / 2, [1])))[:3]
  new_center = np.array(output_shape) / 2
  new_affine[:3, 3] = original_center - new_center
  
  # Create new image
  new_img = nib.Nifti1Image(resized_data, new_affine)
  return new_img

def align(img):
  """ Aligns a given nifti file to this specific affine matrix """
  specific_affine = np.array([
    [1, 0, 0, -128],
    [0, 1, 0, -128],
    [0, 0, 1, -128],
    [0, 0, 0, 1]])
  return nib.Nifti1Image(img.get_fdata(), specific_affine)

if __name__ == "__main__":
  input_img = nib.load(input("Path to nifti file: "))
  # Apply the conform function
  output_img = align(conform(input_img))
  # Save the result
  nib.save(output_img, input("Path to output: "))
