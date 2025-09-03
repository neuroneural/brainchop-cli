from pydawn import utils, webgpu
import numpy as np

def compute_dot_product(vec1, vec2):
    """Compute dot product of two vectors using GPU."""
    n = len(vec1)
    
    # Initialize GPU
    adapter = utils.request_adapter_sync(
        power_preference=webgpu.WGPUPowerPreference_HighPerformance
    )
    dev = utils.request_device_sync(adapter)
    
    # Shader: each thread computes one element product, then sum
    shader_source = f"""
        @group(0) @binding(0)
        var<storage,read> vec1: array<f32>;
        
        @group(0) @binding(1)
        var<storage,read> vec2: array<f32>;
        
        @group(0) @binding(2)
        var<storage,read_write> products: array<f32>;
        
        @compute
        @workgroup_size(1)
        fn main(@builtin(global_invocation_id) id: vec3<u32>) {{
            let i = id.x;
            if (i < {n}u) {{
                products[i] = vec1[i] * vec2[i];
            }}
        }}
    """
    
    shader_module = utils.create_shader_module(dev, shader_source)
    
    # Create buffers
    vec1_np = np.array(vec1, dtype=np.float32)
    vec2_np = np.array(vec2, dtype=np.float32)
    
    buffer1 = utils.create_buffer(
        dev, vec1_np.nbytes, 
        webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst
    )
    utils.write_buffer(dev, buffer1, 0, bytearray(vec1_np.tobytes()))
    
    buffer2 = utils.create_buffer(
        dev, vec2_np.nbytes,
        webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopyDst
    )
    utils.write_buffer(dev, buffer2, 0, bytearray(vec2_np.tobytes()))
    
    result_buffer = utils.create_buffer(
        dev, vec1_np.nbytes,
        webgpu.WGPUBufferUsage_Storage | webgpu.WGPUBufferUsage_CopySrc
    )
    
    # Setup pipeline
    binding_layouts = [
        {
            "binding": 0,
            "visibility": webgpu.WGPUShaderStage_Compute,
            "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage},
        },
        {
            "binding": 1,
            "visibility": webgpu.WGPUShaderStage_Compute,
            "buffer": {"type": webgpu.WGPUBufferBindingType_ReadOnlyStorage},
        },
        {
            "binding": 2,
            "visibility": webgpu.WGPUShaderStage_Compute,
            "buffer": {"type": webgpu.WGPUBufferBindingType_Storage},
        },
    ]
    
    bindings = [
        {"binding": 0, "resource": {"buffer": buffer1, "offset": 0, "size": vec1_np.nbytes}},
        {"binding": 1, "resource": {"buffer": buffer2, "offset": 0, "size": vec2_np.nbytes}},
        {"binding": 2, "resource": {"buffer": result_buffer, "offset": 0, "size": vec1_np.nbytes}},
    ]
    
    bind_group_layout = utils.create_bind_group_layout(device=dev, entries=binding_layouts)
    pipeline_layout = utils.create_pipeline_layout(device=dev, bind_group_layouts=[bind_group_layout])
    bind_group = utils.create_bind_group(device=dev, layout=bind_group_layout, entries=bindings)
    
    compute_pipeline = utils.create_compute_pipeline(
        device=dev,
        layout=pipeline_layout,
        compute={"module": shader_module, "entry_point": "main"},
    )
    
    # Execute
    command_encoder = utils.create_command_encoder(dev)
    compute_pass = utils.begin_compute_pass(command_encoder)
    
    utils.set_pipeline(compute_pass, compute_pipeline)
    utils.set_bind_group(compute_pass, bind_group)
    utils.dispatch_workgroups(compute_pass, n, 1, 1)
    
    utils.end_compute_pass(compute_pass)
    cb_buffer = utils.command_encoder_finish(command_encoder)
    utils.submit(dev, [cb_buffer])
    
    # Read products and sum on CPU
    byte_array = utils.read_buffer(dev, result_buffer)
    products = np.frombuffer(byte_array, dtype=np.float32)
    return products.sum()


if __name__ == "__main__":
    vec1 = [1.0, 2.0, 3.0, 4.0]
    vec2 = [5.0, 6.0, 7.0, 8.0]
    
    result = compute_dot_product(vec1, vec2)
    expected = np.dot(vec1, vec2)
    
    print(f"GPU result: {result}")
    print(f"Expected: {expected}")
    print(f"Match: {np.isclose(result, expected)}")
