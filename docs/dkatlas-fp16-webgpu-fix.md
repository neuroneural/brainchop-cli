# DKatlas FP16 WebGPU Export Fix

## Problem

DKatlas model (104 output classes) was failing to run in WebGPU due to buffer size limits. The model's intermediate activations exceeded the 4GB WebGPU buffer limit when using fp32:

```
Buffer size (6979321856) exceeds the max buffer size limit (4294967200).
```

The calculation: 104 classes × 256³ voxels × 4 bytes = ~6.9GB

## Solution

Convert DKatlas to use fp16 (half precision) for weights and activations, reducing memory by 50%.

## Issues Encountered and Fixes

### 1. Empty Safetensors File (Original Issue)

**Symptom:** DKatlas safetensors was only 16 bytes (empty).

**Root Cause:** DKatlas was using the tfjs backend which stores weights in `.bin` format. The tfjs backend's `ModelContainer` doesn't expose weights to tinygrad's `get_state_dict()`.

**Fix:** Created `convert_tfjs_to_pth.py` to convert tfjs weights to tinygrad format, and added DKatlas to `NEW_BACKEND` set in `utils.py`.

### 2. Weights Not Exported to WebGPU JS

**Symptom:** After fp16 conversion, output was 100% Label 0. The exported JS had no `createWeightBuf` calls - all buffers were `createEmptyBuf`.

**Root Cause:** The `.half()` method created lazy CAST operations that weren't realized. This meant `tensor.uop.base.realized` was `None` for all weights. Since `id(None)` is the same for all tensors, the export couldn't match weight tensors to their names in the state dict.

**Fix:** Added `.realize()` after each `.half()` call in `tiny_meshnet.py`:

```python
def half(self):
    for layer in self.model:
        if isinstance(layer, nn.Conv2d):
            layer.weight = layer.weight.half().realize()  # Added .realize()
            if layer.bias is not None:
                layer.bias = layer.bias.half().realize()
    return self
```

### 3. WebGPU Buffer Size Alignment

**Symptom:** `RangeError: createBuffer failed, size (1134) is not a multiple of 4 when mappedAtCreation == true`

**Root Cause:** WebGPU requires buffer sizes to be multiples of 4 bytes when `mappedAtCreation` is true. FP16 weights can have odd byte counts (e.g., 567 fp16 values = 1134 bytes).

**Fix:** Added padding in `export_model.py`:

```javascript
const createWeightBuf = (device, size, data) => {
  const paddedSize = Math.ceil(size / 4) * 4;
  const buf = device.createBuffer({ size: paddedSize, ... });
  ...
};
```

### 4. Browser Float16Array Support

**Consideration:** The exported model expects `Float16Array` input. This is natively supported in modern browsers since June 2024 (TC39 Stage 4). For the niivue-tinygrad frontend, added conversion:

```javascript
if (selectedModel['fp16']) {
  inputData = new Float16Array(img32);
}
```

## Files Modified

### brainchop-cli
- `brainchop/tiny_meshnet.py` - Added `half()` method with `.realize()` calls, FP16 env var support
- `brainchop/main.py` - Added FP16 env var check for input tensor casting
- `brainchop/export_model.py` - Added buffer size padding for WebGPU alignment
- `brainchop/utils.py` - Added "DKatlas" to NEW_BACKEND set
- `brainchop/convert_tfjs_to_pth.py` - New script for tfjs to pth conversion

### niivue-tinygrad
- `main.js` - Added `fp16: true` config flag and Float16Array conversion
- `net_DKatlas.js` - Re-exported with fp16 support
- `public/net_DKatlas.safetensors` - Converted weights (174KB vs 347KB fp32)

## Usage

To export a model with fp16:

```bash
FP16=1 WEBGPU=1 PREARGMAX=1 EXPORT=1 brainchop -m DKatlas input.nii.gz -o output.nii.gz
```

## Result

- Weight file size: 174KB (was 347KB in fp32)
- JS file size: 140KB
- Buffer requirements reduced by ~50%
- Model now runs within WebGPU's 4GB limit
