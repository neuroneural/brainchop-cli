import { Niivue } from '@niivue/niivue'
// IMPORTANT: we need to import this specific file. 
import subcortical from "./net_subcortical.js"
import tissue_fast from "./net_tissue_fast.js"
import mindgrab from "./net_mindgrab.js"
import t2 from "./net_t2.js"

const models = {
  "subcortical": {
    "net": subcortical,
    "weightPath": "./net_subcortical.safetensors",
    "colormap": "./colormap_tissue_subcortical.json",
    "volume": "./t1_crop.nii.gz"
  },
  "tissue_fast": {
    "net": tissue_fast,
    "weightPath":
    "./net_tissue_fast.safetensors",
    "colormap": "./colormap_tissue_subcortical.json",
    "volume": "./t1_crop.nii.gz"
  },
  "mindgrab": {
    "net": mindgrab,
    "weightPath": "./net_mindgrab.safetensors",
    "colormap": "./colormap_tissue_subcortical.json",
    "volume": "./t1_crop.nii.gz"
  },
  "t2": {
    "net": t2,
    "weightPath":
    "./net_t2.safetensors",
    "colormap": "./colormap_t2.json",
    "volume": "./M2265_T2w.nii.gz"
  }
}

let selectedModel = models[document.getElementById("segmentationDropdown").value]

async function main() {
  clipCheck.onchange = function () {
    if (clipCheck.checked) {
      nv1.setClipPlane([0, 0, 90])
    } else {
      nv1.setClipPlane([2, 0, 90])
    }
  }
  opacitySlider0.oninput = function () {
    nv1.setOpacity(0, opacitySlider0.value / 255)
    nv1.updateGLVolume()
  }
  opacitySlider1.oninput = function () {
    nv1.setOpacity(1, opacitySlider1.value / 255)
  }
  function doLoadImage() {
    opacitySlider0.oninput()
  }
  async function fetchJSON(fnm) {
    const response = await fetch(fnm)
    const js = await response.json()
    return js
  }
  saveImgBtn.onclick = function () {
    nv1.volumes[1].saveToDisk('Custom.nii')
  }
  aboutBtn.onclick = function () {
    const url = "https://github.com/niivue/niivue-tinygrad"
    window.open(url, "_blank")
  }
  async function ensureConformed() {
    const nii = nv1.volumes[0]
    let isConformed = nii.dims[1] === 256 && nii.dims[2] === 256 && nii.dims[3] === 256
    if (nii.permRAS[0] !== -1 || nii.permRAS[1] !== 3 || nii.permRAS[2] !== -2) {
      isConformed = false
    }
    if (isConformed) {
      return
    }
    const nii2 = await nv1.conform(nii, false)
    await nv1.removeVolume(nv1.volumes[0])
    await nv1.addVolume(nii2)
  }
  async function closeAllOverlays() {
    while (nv1.volumes.length > 1) {
      await nv1.removeVolume(nv1.volumes[1])
    }
  }
  const getDevice = async () => {
    if (!navigator.gpu) return false;
    const requiredLimits = {};
    const maxBufferSize = 1409286144;
    requiredLimits.maxStorageBufferBindingSize = maxBufferSize;
    requiredLimits.maxBufferSize = maxBufferSize;
    const adapter = await navigator.gpu.requestAdapter();
    return await adapter.requestDevice({
        requiredLimits: requiredLimits,
        requiredFeatures: ["shader-f16"]
    });
  };

  const device = await getDevice();

  function convertInMemoryOrder(inverse, size, data) {
    let output = new Float32Array(data.length)
    let it = 0;

    for (let x = 0; x < size; x++) {
      for (let y = 0; y < size; y++) {
        for (let z = 0; z < size; z++) {
          let idx = x + y * size + z * size * size;
          if (inverse) {
            output[idx] = data[it++];
          } else {
            output[it++] = data[idx];
          }
        }
      }
    }

    return output
  }

  segmentBtn.onclick = async function () {
    if (nv1.volumes.length < 1) {
      window.alert('Please open a voxel-based image')
      return
    }
    const startTime = Date.now();
    loadingCircle.classList.remove('hidden')
    await closeAllOverlays()
    await ensureConformed()

    let img32 = convertInMemoryOrder(/*inverse*/ false, 256, new Float32Array(nv1.volumes[0].img))

    // normalize input data to range 0..1
    let mx = img32[0]
    let mn = mx
    for (let i = 0; i < img32.length; i++) {
      mx = Math.max(mx, img32[i])
      mn = Math.min(mn, img32[i])
    }
    let scale32 = 1 / (mx - mn)
    for (let i = 0; i < img32.length; i++) {
      img32[i] = (img32[i] - mn) * scale32
    }
  
    const session = await selectedModel["net"].load(device, selectedModel["weightPath"]);

    const shape = [1, 1, 256, 256, 256]
    const nvox = shape.reduce((a, b) => a * b)
    if (img32.length !== nvox) {
      throw new Error(`img32 length (${img32.length}) does not match expected tensor length (${expectedLength})`)
    }

    const results = await session(img32);
   
    let segmentImg = nv1.cloneVolume(0)
    segmentImg.img = convertInMemoryOrder(/*inverse*/ true, 256, results[0])
    segmentImg.hdr.datatypeCode = 16
    segmentImg.hdr.dims[4] = 1
    segmentImg.trustCalMinMax = false

    // Add the output to niivue
    const cmap = await fetchJSON(selectedModel["colormap"])
    segmentImg.setColormapLabel(cmap)
    segmentImg.opacity = opacitySlider1.value / 255
    await nv1.addVolume(segmentImg)
    loadingCircle.classList.add('hidden')
    const elapsedTime = Date.now() - startTime
    document.getElementById("intensity").innerHTML = `${elapsedTime}ms to segment`
  }
  function handleLocationChange(data) {
    document.getElementById("intensity").innerHTML = data.string
  }
  const defaults = {
    backColor: [0.4, 0.4, 0.4, 1],
    onLocationChange: handleLocationChange,
  }
  const nv1 = new Niivue(defaults)
  nv1.attachToCanvas(gl1)
  nv1.opts.multiplanarForceRender = true
  nv1.opts.yoke3Dto2DZoom = true
  nv1.opts.crosshairGap = 11
  nv1.setInterpolation(true)
  nv1.onImageLoaded = doLoadImage
  await nv1.loadVolumes([{ url: selectedModel["volume"] }])
  segmentBtn.onclick()

  document.getElementById("segmentationDropdown").addEventListener("change", async function () {
    selectedModel = models[this.value]
    if (nv1.volumes[0].url != selectedModel["volume"]) {
      nv1.removeVolumeByIndex(0)
      await nv1.loadVolumes([{ url: selectedModel["volume"] }])
    }
    segmentBtn.onclick()
  });
}

main()
