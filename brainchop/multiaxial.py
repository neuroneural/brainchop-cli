import os
import numpy as np
import nibabel as nib
from tinygrad.tensor import Tensor
from skimage.transform import resize
from nibabel.orientations import axcodes2ornt, ornt_transform
from tinygrad.helpers import tqdm
import time
from .tinyonnx import OnnxRunner


# ---------------------------
# Image Orientation Functions
# ---------------------------
def reorient(nii, orientation) -> nib.Nifti1Image:
    """Reorients a nifti image to specified orientation."""
    orig_ornt = nib.io_orientation(nii.affine)
    targ_ornt = axcodes2ornt(orientation)
    transform = ornt_transform(orig_ornt, targ_ornt)
    reoriented_nii = nii.as_reoriented(transform)
    return reoriented_nii

def create_coordinate_matrix(shape, anterior_commissure):
    """Creates a coordinate matrix based on the image shape and anterior commissure."""
    x, y, z = shape
    meshgrid = np.meshgrid(np.linspace(0, x - 1, x), np.linspace(0, y - 1, y), np.linspace(0, z - 1, z), indexing='ij')
    coordinates = np.stack(meshgrid, axis=-1) - np.array(anterior_commissure)
    matrix_with_ones = np.concatenate([coordinates, np.ones((coordinates.shape[0], coordinates.shape[1], coordinates.shape[2], 1))], axis=-1)
    return matrix_with_ones

# ----------------------
# Preprocessing Functions
# ----------------------
def preprocess_head_MRI(nii, anterior_commissure=None, keep_parameters_for_reconstruction=False):
    """Preprocesses a head MRI image."""    
    if anterior_commissure is None:
        print('No anterior commissure location given.. centering to center of image..')
        anterior_commissure = [nii.shape[0]//2, nii.shape[1]//2, nii.shape[2]//2]
    else:
        print(f'anterior commissure given: {anterior_commissure}')
    orientation = nib.aff2axcodes(nii.affine)
    
    if ''.join(orientation) != 'RAS':
        print(f'Image orientation : {orientation}. Changing to RAS..')
        nii = reorient(nii, "RAS")
                
    # Make image isotropic
    res = nii.header['pixdim'][1:4]
    img = nii.get_fdata()
    new_shape = np.array(np.array(nii.shape)*res, dtype='int')
    if np.any(np.array(nii.shape) != new_shape):
        img = resize(img, new_shape, anti_aliasing=True, preserve_range=True)
       
    nii.affine[0][0] = 1.
    nii.affine[1][1] = 1.
    nii.affine[2][2] = 1.
    nii.header['pixdim'][1:4] = np.diag(nii.affine)[0:3]
    
    # Crop/Pad to make shape 256,256,256
    d1, d2, d3 = new_shape
    start = None
    end = None    
    
    if d1 < 256:
        pad1 = 256-d1
        img = np.pad(img, ((pad1//2, pad1//2+pad1%2),(0,0),(0,0)))
        anterior_commissure[0] += pad1//2
    
    if d2 > 256: 
        crop2 = d2-256
        img = img[:,crop2//2:-(crop2//2+crop2%2)]
        anterior_commissure[1] -= crop2//2
            
    elif d2 < 256:
        pad2 = 256-d2
        img = np.pad(img, ((0,0),(pad2//2, pad2//2+pad2%2),(0,0)))
        anterior_commissure[1] += pad2//2
        
    if d3 > 256: 
        # Head start
        proj = np.max(img,(0,1))
        proj[proj < np.percentile(proj, 50) ] = 0
        proj[proj > 0] = 1
        end = np.max(np.argwhere(proj == 1))
        end = np.min([end + 20, d3]) # Leave some space above the head
        start = end-256
        if start < 0:
            crop3 = d3 - 256
            img = img[:,:,crop3:]
            anterior_commissure[2] -= crop3
         
        else:
            img = img[:,:,start:end]
            anterior_commissure[2] -= start
    elif d3 < 256:
        pad3 = 256-d3
        img = np.pad(img, ((0,0),(0,0),(pad3//2, pad3//2+pad3%2)))
        anterior_commissure[2] += pad3//2
    
    coords = create_coordinate_matrix(img.shape, anterior_commissure)        
    
    # Intensity normalization
    p95 = np.percentile(img, 95)
    img = img/p95
    
    coords = coords[:,:,:,:3]
    coords = coords/256.
    
    result = [nib.Nifti1Image(img, nii.affine)]
        
    result.extend([np.array(coords, dtype='float32'), np.array(anterior_commissure, dtype='int')])
    
    if keep_parameters_for_reconstruction:
        reconstruction_parms = d1, d2, d3, start, end
        result.append(reconstruction_parms)
    
    return tuple(result)

# ------------------
# TinyGrad ONNX Segmentation Logic
# ------------------
def process_slices(runner, img, coords, axis=0, batch_size=2, input_names=None):
    # Initialize output array (7 classes per slice prediction)
    output = np.zeros((img.shape[0], img.shape[1], img.shape[2], 7), dtype=np.float32)
    
    if input_names is None:
        input_names = {"img": "input_1", "coords": "input_2"}
    
    img_input_name = input_names["img"]
    coords_input_name = input_names["coords"]
    
    num_slices = img.shape[axis]
    all_img_batches = []
    all_coords_batches = []
    all_batch_indices = []

    print('prep batch')
    # Prepare all input batches
    for batch_start in range(0, num_slices, batch_size):
        batch_end = min(batch_start + batch_size, num_slices)
        batch_size_actual = batch_end - batch_start
        
        if axis == 0:
            img_batch = img[batch_start:batch_end, :, :].reshape(batch_size_actual, img.shape[1], img.shape[2], 1)
            coords_batch = coords[batch_start:batch_end, :, :, :]
        elif axis == 1:
            img_batch = img[:, batch_start:batch_end, :].transpose(1, 0, 2).reshape(batch_size_actual, img.shape[0], img.shape[2], 1)
            coords_batch = coords[:, batch_start:batch_end, :, :].transpose(1, 0, 2, 3)
        else:
            img_batch = img[:, :, batch_start:batch_end].transpose(2, 0, 1).reshape(batch_size_actual, img.shape[0], img.shape[1], 1)
            coords_batch = coords[:, :, batch_start:batch_end, :].transpose(2, 0, 1, 3)
        
        all_img_batches.append(img_batch.astype(np.float32))
        all_coords_batches.append(coords_batch.astype(np.float32))
        all_batch_indices.append((batch_start, batch_end))

    print('inference')
    # Create inputs and run inference
    all_outputs = []
    for img_batch, coords_batch in zip(all_img_batches, all_coords_batches):
        model_input = {
            img_input_name: Tensor(img_batch),
            coords_input_name: Tensor(coords_batch)
        }
        all_outputs.append(runner(model_input))

    print('stitch')
    # Process outputs
    output_tensors = [list(out.values())[0] for out in all_outputs]
    
    # Concatenate along batch dimension
    full_output = Tensor.cat(*output_tensors, dim=0).numpy()

    # Apply axis-specific permutation
    if axis == 1:
        full_output = np.transpose(full_output, (1, 0, 2, 3))
    elif axis == 2:
        full_output = np.transpose(full_output, (1, 2, 0, 3))

    return output

def get_input_names(model):
    """Extract input names from an ONNX model."""
    input_names = {}
    for i, input_info in enumerate(model.graph.input):
        if i == 0:
            input_names["img"] = input_info.name
        elif i == 1:
            input_names["coords"] = input_info.name
    return input_names

def extract_weights_from_onnx(model_path):
    import onnx
    from onnx import numpy_helper
    
    model = onnx.load(model_path)
    
    # Find the convolution weights and biases
    weights = None
    biases = None
    
    for initializer in model.graph.initializer:
        if initializer.name == "model/conv3d/Conv3D/ReadVariableOp:0":
            weights = numpy_helper.to_array(initializer)
        elif initializer.name == "model/conv3d/BiasAdd/ReadVariableOp:0":
            biases = numpy_helper.to_array(initializer)
    return weights, biases

def optimized_consensus(combined_data, weights, biases):
    height, width, depth, channels = combined_data.shape
    
    # Reshape weights to [22, 7] for matrix multiplication
    weights_2d = weights.reshape(7, channels).transpose()
    
    # Reshape input to [voxels, channels]
    flat_data = combined_data.reshape(-1, channels)
    
    # Process in batches for memory efficiency
    batch_size = 1000000  # Adjust based on available memory
    total_voxels = flat_data.shape[0]
    output_flat = np.zeros(total_voxels, dtype=np.int64)
    
    
    for start_idx in tqdm(range(0, total_voxels, batch_size)):
        end_idx = min(start_idx + batch_size, total_voxels)
        batch = flat_data[start_idx:end_idx]
        
        # Apply matrix multiplication: [batch, channels] × [channels, 7] = [batch, 7]
        logits = np.matmul(batch, weights_2d)
        
        # Add biases
        logits += biases
        
        # Get class predictions (argmax)
        predictions = np.argmax(logits, axis=1)
        
        # Store predictions
        output_flat[start_idx:end_idx] = predictions
    
    # Reshape back to volume
    output = output_flat.reshape(height, width, depth)
    
    return output

def multiaxial_segmentation(img, model_dir):
    """
    Perform multiaxial segmentation using the ONNX models
    
    Args:
        img: Image as nibabel Nifti1Image
        model_dir: Base directory for the ONNX models
        
    Returns:
        numpy.ndarray: Segmentation output
    """
    import onnx
    
    # Define model paths
    sagittal_model_path = os.path.join(model_dir, "sagittal_model.onnx")
    axial_model_path = os.path.join(model_dir, "axial_model.onnx")
    coronal_model_path = os.path.join(model_dir, "coronal_model.onnx")
    consensus_model_path = os.path.join(model_dir, "consensus_layer.onnx")
    
    # Check if model files exist
    for model_path in [sagittal_model_path, axial_model_path, coronal_model_path, consensus_model_path]:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
    
    
    nii_out, coords, anterior_commissure, reconstruction_parms = preprocess_head_MRI(
        img, 
        anterior_commissure=None,
        keep_parameters_for_reconstruction=True
    )
    
    # Load models
    view_models = [
        (sagittal_model_path, 0, "sagittal"),  # axis 0 = sagittal
        (coronal_model_path, 1, "coronal"),    # axis 1 = coronal
        (axial_model_path, 2, "axial")         # axis 2 = axial
    ]
    
    view_outputs = [None, None, None]
    img_data = nii_out.get_fdata()
    
    for i, (model_path, axis, name) in enumerate(view_models):
        if model_path is not None:
            print(f"Running {name} model inference...")
            model = onnx.load(model_path)
            runner = OnnxRunner(model)
            input_names = get_input_names(model)
            view_outputs[i] = process_slices(
                runner, img_data, coords, axis=axis, input_names=input_names
            )
    
    # Create empty outputs for any models that didn't run
    for i in range(3):
        if view_outputs[i] is None:
            view_outputs[i] = np.zeros((img_data.shape[0], img_data.shape[1], img_data.shape[2], 7), dtype=np.float32)
    
    # Extract results
    model_segmentation_sagittal = view_outputs[0]
    model_segmentation_coronal = view_outputs[1]
    model_segmentation_axial = view_outputs[2]
    
    # Extract consensus model weights
    print("Extracting consensus model weights and biases...")
    weights, biases = extract_weights_from_onnx(consensus_model_path)
    
    # Prepare input for consensus model
    img_expanded = np.expand_dims(img_data, -1).astype(np.float32)
    
    combined_data = np.concatenate([
        img_expanded, 
        model_segmentation_sagittal,
        model_segmentation_coronal,
        model_segmentation_axial
    ], axis=-1)
    output = optimized_consensus(combined_data, weights, biases)
    return output
