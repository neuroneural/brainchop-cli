import argparse
from nibabel import save, load, Nifti1Image
import numpy as np
from tinygrad import Tensor
from .model import tinygrad_model

def main():
  parser = argparse.ArgumentParser(description="BrainChop: Brain segmentation tool")
  parser.add_argument("input", help="Input NIfTI file path")
  parser.add_argument("-o", "--output", default="output.nii.gz", help="Output NIfTI file path")
  parser.add_argument("--model", default="example", help="Name of segmentation model")
  parser.add_argument("--json", default="./example/model.json", help="Path to model JSON file")
  parser.add_argument("--bin", default="./example/model.bin", help="Path to model binary file")
  
  args = parser.parse_args()

  img = load(args.input)
  tensor = np.array(img.dataobj).reshape(1, 1, 256, 256, 256)
  t = Tensor(tensor.astype(np.float16))
  out_tensor = tinygrad_model(args.json, args.bin, t)
  save(Nifti1Image(out_tensor, img.affine, img.header), args.output)
  print(f"Output saved as {args.output}")

if __name__ == "__main__":
  main()
