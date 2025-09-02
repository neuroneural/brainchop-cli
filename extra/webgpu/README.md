### niivue-tinygrad

[tinygrad](https://tinygrad.org/) WebGPU live demo using [brainchop](https://github.com/neuroneural/brainchop) models to segment a T1-scan into gray and white matter This project mimics the [ONNX](https://github.com/niivue/niivue-onnx) WebGPU and [brainchop](https://github.com/neuroneural/brainchop) TensorFlowJS WebGL projects. However, tinygrad has a smaller footprint and is faster. This port developed by Ahmed Harmouche of [softwiredtech](https://github.com/softwiredtech).

### For Developers

You can serve a hot-reloadable web page that allows you to interactively modify the source code.

```bash
git clone git@github.com:spikedoanz/niivue-tinygrad.git
cd niivue-tinygrad
npm install
npm run dev
```

#### to build and serve the built version

```bash
npm run build
npx http-server dist/ # or npm run preview
```

