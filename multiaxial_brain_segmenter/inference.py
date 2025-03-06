#!/usr/bin/env python3
import os
import sys
import nibabel as nib
import numpy as np
from tqdm import tqdm
import time
from skimage.transform import resize
from nibabel.orientations import axcodes2ornt, ornt_transform
from tinygrad.tensor import Tensor
from tinyonnx import OnnxRunner

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
def preprocess_head_MRI(nii: nib.Nifti1Image, nii_seg: nib.Nifti1Image = None, anterior_commissure: tuple = None, keep_parameters_for_reconstruction: bool = False):
    """Preprocesses a head MRI image."""    
    if nii_seg is not None:
        assert nii.shape == nii_seg.shape
    if anterior_commissure is None:
        print('No anterior commissure location given.. centering to center of image..')
        anterior_commissure = [nii.shape[0]//2, nii.shape[1]//2, nii.shape[2]//2]
    else:
        print(f'anterior commissure given: {anterior_commissure}')
    orientation = nib.aff2axcodes(nii.affine)
    
    if ''.join(orientation) != 'RAS':
        print(f'Image orientation : {orientation}. Changing to RAS..')
        nii = reorient(nii, "RAS")
        if nii_seg is not None:
            nii_seg = reorient(nii_seg, "RAS")
   
    if nii_seg is not None:
        img_seg = nii_seg.get_fdata()
    else:
        print('Segmentation image not given')
                
    # Make image isotropic
    res = nii.header['pixdim'][1:4]
    img = nii.get_fdata()
    new_shape = np.array(np.array(nii.shape)*res, dtype='int')
    if np.any(np.array(nii.shape) != new_shape):
        img = resize(img, new_shape, anti_aliasing=True, preserve_range=True)
        if nii_seg is not None: 
            img_seg = resize(img_seg, new_shape, order=0, anti_aliasing=True, preserve_range=True)
       
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
        if nii_seg is not None: 
            img_seg = np.pad(img_seg, ((pad1//2, pad1//2+pad1%2),(0,0),(0,0)))
        anterior_commissure[0] += pad1//2
    
    if d2 > 256: 
        crop2 = d2-256
        img = img[:,crop2//2:-(crop2//2+crop2%2)]
        if nii_seg is not None: 
            img_seg = img_seg[:,crop2//2:-(crop2//2+crop2%2)]
        anterior_commissure[1] -= crop2//2
            
    elif d2 < 256:
        pad2 = 256-d2
        img = np.pad(img, ((0,0),(pad2//2, pad2//2+pad2%2),(0,0)))
        if nii_seg is not None: 
            img_seg = np.pad(img_seg, ((0,0),(pad2//2, pad2//2+pad2%2),(0,0)))
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
            if nii_seg is not None: 
                img_seg = img_seg[:,:,crop3:]
            anterior_commissure[2] -= crop3
         
        else:
            img = img[:,:,start:end]
            if nii_seg is not None: 
                img_seg = img_seg[:,:,start:end]
            anterior_commissure[2] -= start
    elif d3 < 256:
        pad3 = 256-d3
        img = np.pad(img, ((0,0),(0,0),(pad3//2, pad3//2+pad3%2)))
        if nii_seg is not None: 
            img_seg = np.pad(img_seg, ((0,0),(0,0),(pad3//2, pad3//2+pad3%2)))
        anterior_commissure[2] += pad3//2
    
    coords = create_coordinate_matrix(img.shape, anterior_commissure)        
    
    # Intensity normalization
    p95 = np.percentile(img, 95)
    img = img/p95
    
    coords = coords[:,:,:,:3]
    coords = coords/256.
    
    result = [nib.Nifti1Image(img, nii.affine)]
    
    if nii_seg is not None:
        result.append(nib.Nifti1Image(np.array(img_seg, dtype='int8'), nii.affine))
    else:
        result.append(None)
        
    result.extend([np.array(coords, dtype='float32'), np.array(anterior_commissure, dtype='int')])
    
    if keep_parameters_for_reconstruction:
        reconstruction_parms = d1, d2, d3, start, end
        result.append(reconstruction_parms)
    
    return tuple(result)
        
def reshape_back_to_original(img, nii_original, reconstruction_parms, resample_order=1):
    """Reshape a processed image back to its original dimensions."""
    d1, d2, d3, start, end = reconstruction_parms
    
    # Pad or crop z axis
    if d3 > 256: 
        if start < 0:
            crop3 = d3 - 256
            img = np.pad(img, ((0,0),(0,0),(crop3,0)))
        else:
            img = np.pad(img, ((0,0),(0,0),(start,end)))          
    elif d3 < 256:
        pad3 = 256-d3
        img = img[:,:,pad3//2:-(pad3//2+pad3%2)]
    
    # Pad or crop y axis
    if d2 > 256: 
        crop2 = d2-256
        img = np.pad(img, ((0,0),(crop2//2,crop2//2+crop2%2),(0,0)))
    elif d2 < 256:
        pad2 = 256-d2
        img = img[:,pad2//2:-(pad2//2+pad2%2),:]
        
    # Pad or crop x axis
    if d1 < 256:
        pad1 = 256-d1
        img = img[pad1//2:-(pad1//2+pad1%2),:,:]
    
    # Resample to original resolution
    img = resize(img, nii_original.shape, anti_aliasing=True, preserve_range=True, order=resample_order)
    
    nii_out = nib.Nifti1Image(img, nii_original.affine)
    return nii_out

# ------------------
# TinyGrad ONNX Segmentation Logic
# ------------------
def process_slices(runner, img, coords, axis=0, input_names=None):
    """
    Process slices through the model and return the predictions.
    
    Args:
        runner: OnnxRunner instance
        img: The preprocessed image data
        coords: The coordinate data
        axis: Axis to slice along (0=sagittal, 1=coronal, 2=axial)
        input_names: Dictionary mapping input types to model input names
    
    Returns:
        Predictions as a 4D numpy array [height, width, depth, classes]
    """
    # Initialize output array (7 classes per slice prediction)
    output = np.zeros((img.shape[0], img.shape[1], img.shape[2], 7), dtype=np.float32)
    
    # Map axis numbers to names for logging
    axis_names = {0: "sagittal", 1: "coronal", 2: "axial"}
    print(f"Processing {img.shape[axis]} {axis_names[axis]} slices...")
    
    # Default input names if not provided
    if input_names is None:
        input_names = {"img": "input_1", "coords": "input_2"}
    
    img_input_name = input_names["img"]
    coords_input_name = input_names["coords"]
    
    # Process each slice
    for i in tqdm(range(img.shape[axis])):
        # Extract 2D slice based on the specified axis
        if axis == 0:  # Sagittal (YZ plane)
            img_slice = img[i, :, :]
            coords_slice = coords[i, :, :, :]
        elif axis == 1:  # Coronal (XZ plane)
            img_slice = img[:, i, :]
            coords_slice = coords[:, i, :, :]
        else:  # Axial (XY plane)
            img_slice = img[:, :, i]
            coords_slice = coords[:, :, i, :]
        
        # Prepare inputs with correct shapes
        img_input = np.expand_dims(np.expand_dims(img_slice, -1), 0).astype(np.float32)
        coords_input = np.expand_dims(coords_slice, 0).astype(np.float32)
        
        # Create TinyGrad tensors
        img_tensor = Tensor(img_input, requires_grad=False)
        coords_tensor = Tensor(coords_input, requires_grad=False)
        
        # Run inference with correct input names
        outputs = runner({
            img_input_name: img_tensor, 
            coords_input_name: coords_tensor
        })
        
        # Get output tensor and convert to numpy
        output_tensor = list(outputs.values())[0]
        
        # Store prediction according to the correct axis
        if axis == 0:  # Sagittal
            output[i, :, :, :] = output_tensor.numpy()[0]
        elif axis == 1:  # Coronal
            output[:, i, :, :] = output_tensor.numpy()[0]
        else:  # Axial
            output[:, :, i, :] = output_tensor.numpy()[0]
    
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
    """
    Extract weights and biases from the consensus layer ONNX model.
    
    Args:
        model_path: Path to the ONNX model file
        
    Returns:
        tuple: (weights, biases)
    """
    import onnx
    from onnx import numpy_helper
    
    model = onnx.load(model_path)
    
    # Find the convolution weights and biases
    weights = None
    biases = None
    
    for initializer in model.graph.initializer:
        if initializer.name == "model/conv3d/Conv3D/ReadVariableOp:0":
            weights = numpy_helper.to_array(initializer)
            print(f"Extracted consensus weights with shape {weights.shape}")
        elif initializer.name == "model/conv3d/BiasAdd/ReadVariableOp:0":
            biases = numpy_helper.to_array(initializer)
            print(f"Extracted consensus biases with shape {biases.shape}")
    
    return weights, biases

def optimized_consensus(combined_data, weights, biases):
    """
    Apply the optimized consensus function using direct matrix multiplication.
    
    Args:
        combined_data: Combined data with shape [height, width, depth, channels]
        weights: Convolution weights with shape [7, 22, 1, 1, 1]
        biases: Biases with shape [7]
        
    Returns:
        numpy.ndarray: Final segmentation with shape [height, width, depth]
    """
    height, width, depth, channels = combined_data.shape
    
    # Reshape weights to [22, 7] for matrix multiplication
    weights_2d = weights.reshape(7, channels).transpose()
    
    # Reshape input to [voxels, channels]
    flat_data = combined_data.reshape(-1, channels)
    
    # Process in batches for memory efficiency
    batch_size = 1000000  # Adjust based on available memory
    total_voxels = flat_data.shape[0]
    output_flat = np.zeros(total_voxels, dtype=np.int64)
    
    print(f"Processing {total_voxels} voxels through optimized consensus...")
    
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

def segment_MRI_tinygrad(img, coords, model_sagittal_path=None, model_axial_path=None, model_coronal_path=None, consensus_model_path=None):
    """
    Perform MRI segmentation using the TinyGrad ONNX runner with optimized consensus.
    
    Args:
        img: The preprocessed image data
        coords: The coordinate data
        model_sagittal_path: Path to the sagittal model ONNX file
        model_axial_path: Path to the axial model ONNX file
        model_coronal_path: Path to the coronal model ONNX file
        consensus_model_path: Path to the consensus model ONNX file
        
    Returns:
        The segmentation output as a 3D numpy array
    """
    # Import necessary models
    import onnx
    
    model_segmentation_sagittal = None
    model_segmentation_coronal = None
    model_segmentation_axial = None
    
    # Process each view model if provided
    view_models = [
        (model_sagittal_path, 0, "sagittal"),  # axis 0 = sagittal
        (model_coronal_path, 1, "coronal"),    # axis 1 = coronal
        (model_axial_path, 2, "axial")         # axis 2 = axial
    ]
    
    view_outputs = [None, None, None]
    
    for i, (model_path, axis, name) in enumerate(view_models):
        if model_path is not None:
            print(f"Running {name} model inference...")
            model = onnx.load(model_path)
            runner = OnnxRunner(model)
            input_names = get_input_names(model)
            view_outputs[i] = process_slices(
                runner, img, coords, axis=axis, input_names=input_names
            )
    
    # Create empty outputs for any models that didn't run
    for i in range(3):
        if view_outputs[i] is None:
            view_outputs[i] = np.zeros((img.shape[0], img.shape[1], img.shape[2], 7), dtype=np.float32)
    
    # Extract results
    model_segmentation_sagittal = view_outputs[0]
    model_segmentation_coronal = view_outputs[1]
    model_segmentation_axial = view_outputs[2]
    
    # Extract consensus model weights
    print("Extracting consensus model weights and biases...")
    weights, biases = extract_weights_from_onnx(consensus_model_path)
    
    # Prepare input for consensus model
    img_expanded = np.expand_dims(img, -1).astype(np.float32)
    
    # Concatenate along the channel dimension
    combined_data = np.concatenate([
        img_expanded, 
        model_segmentation_sagittal,
        model_segmentation_coronal,
        model_segmentation_axial
    ], axis=-1)
    
    print(f"Combined data shape: {combined_data.shape}")
    
    # Apply optimized consensus function
    start_time = time.time()
    output = optimized_consensus(combined_data, weights, biases)
    end_time = time.time()
    
    print(f"Consensus processing completed in {end_time - start_time:.2f} seconds")
    
    return output

# ---------------------
# Main Execution Logic
# ---------------------
def main():
    # Configuration
    OUTPUT_PATH = 'output.nii.gz'
    SCAN_PATH = 'input.nii.gz'
    SAGITTAL_MODEL_PATH = 'models/sagittal_model.onnx'
    AXIAL_MODEL_PATH = 'models/axial_model.onnx'
    CORONAL_MODEL_PATH = 'models/coronal_model.onnx'
    CONSENSUS_LAYER_PATH = 'models/consensus_layer.onnx'
    SEGMENTATION_PATH = None
    ANTERIOR_COMMISSURE = None
    
    # Set working directory to script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)
    
    # Print script information
    print("MRI Segmentation with TinyGrad ONNX Runner (Optimized)")
    print("=====================================================")
    print(f"Working directory: {script_dir}")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Scan path: {SCAN_PATH}")
    print(f"Sagittal model: {os.path.basename(SAGITTAL_MODEL_PATH)}")
    print(f"Axial model: {os.path.basename(AXIAL_MODEL_PATH)}")
    print(f"Coronal model: {os.path.basename(CORONAL_MODEL_PATH)}")
    print(f"Consensus model: {os.path.basename(CONSENSUS_LAYER_PATH)}")
    print("=====================================================\n")
    
    # Load scan
    nii = nib.load(SCAN_PATH)
    nii_seg = nib.load(SEGMENTATION_PATH) if SEGMENTATION_PATH else None
    print(f"Input scan shape: {nii.shape}")
    
    # Extract subject ID
    subject = SCAN_PATH.split('/')[-1].replace('.nii', '')
    if subject.startswith('r'):
        subject = subject[1:]
    
    # Preprocess the MRI
    nii_out, nii_seg_out, coords, anterior_commissure, reconstruction_parms = preprocess_head_MRI(
        nii, 
        nii_seg, 
        anterior_commissure=ANTERIOR_COMMISSURE, 
        keep_parameters_for_reconstruction=True
    )     
    
    # Segment the MRI using TinyGrad's ONNX runtime with optimized consensus
    segmentation = segment_MRI_tinygrad(
        nii_out.get_fdata(), 
        coords, 
        SAGITTAL_MODEL_PATH,
        AXIAL_MODEL_PATH, 
        CORONAL_MODEL_PATH, 
        CONSENSUS_LAYER_PATH
    )
    
    # Save the segmentation result
    nii_out_pred = nib.Nifti1Image(np.array(segmentation, dtype='int16'), nii_out.affine)
    nib.save(nii_out_pred, OUTPUT_PATH)
    print(f"Segmentation saved to {OUTPUT_PATH}")

if __name__ == '__main__':
    main()
