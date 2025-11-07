from pathlib import Path
from tinygrad.tensor import Tensor
from tinygrad.nn.state import get_state_dict, safe_save, safe_load, load_state_dict
from extra.webgpu.export_model import export_model
from tinygrad.helpers import getenv

from brainchop.utils import get_model

if __name__ == "__main__":
  model_name = "mindgrab" 
  mode = "webgpu"
  ext = "js"

  model = get_model(model_name)
  dirname = Path(__file__).parent
  weight_dirname = dirname / "public"
  print(dirname)

  safe_save(get_state_dict(model), (    weight_dirname / f"net_{model_name}.safetensors").as_posix())
  load_state_dict(model, safe_load(str( weight_dirname / f"net_{model_name}.safetensors")))

  input_tensor = Tensor.randn(1,1,256,256,256)
  prg, inp_sizes, out_sizes, state = export_model(model, mode, input_tensor)

  with open(dirname / f"{model_name}.{ext}", "w") as text_file:
    text_file.write(prg)
