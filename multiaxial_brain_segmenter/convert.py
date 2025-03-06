import os
import tensorflow as tf
import tf2onnx
import onnx

def convert_keras_to_onnx(model_path, output_path=None):
    """
    Convert a Keras model to ONNX format.
    
    Args:
        model_path: Path to the Keras .h5 model file
        output_path: Path where to save the ONNX model (default: same as model_path with .onnx extension)
    
    Returns:
        Path to the converted ONNX model
    """
    if output_path is None:
        output_path = model_path.replace('.h5', '.onnx')
    
    # Load the Keras model
    keras_model = tf.keras.models.load_model(model_path, compile=False)
    
    # Get input and output names
    input_signature = []
    for inp in keras_model.inputs:
        shape = inp.shape.as_list()
        # Replace None with a concrete batch size (1)
        shape = [1 if dim is None else dim for dim in shape]
        input_signature.append(tf.TensorSpec(shape, inp.dtype, name=inp.name))
    print(input_signature)
    
    # Convert to ONNX using concrete input shapes
    model_proto, _ = tf2onnx.convert.from_keras(
        keras_model, 
        input_signature=input_signature,
        opset=13  # Use a recent opset version for better compatibility
    )
    
    # Save the ONNX model
    onnx.save_model(model_proto, output_path)
    print(f"Converted {model_path} to {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Example usage
    models_dir = "models"
    
    # Convert all models
    convert_keras_to_onnx(os.path.join(models_dir, "sagittal_model.h5"))
    convert_keras_to_onnx(os.path.join(models_dir, "coronal_model.h5"))
    convert_keras_to_onnx(os.path.join(models_dir, "axial_model.h5"))
    convert_keras_to_onnx(os.path.join(models_dir, "consensus_layer.h5"))
