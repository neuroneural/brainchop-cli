"""
convenience printers for manual visual inspection of brainchop-cli output
"""

import hashlib
from typing import Any
from pathlib import Path
from brainchop.utils import load_models
from tinygrad.helpers import fetch, getenv
import subprocess

CACHEDIR = Path.home() / ".cache" / "brainchop" / "output"
CACHEDIR.mkdir(parents=True, exist_ok=True)

_URLS = {
    "t1_crop": "https://github.com/neuroneural/brainchop-models/raw/main/t1_crop.nii.gz"
}

_MODELS_JSON: dict[str,Any] = load_models()
_MODELS = sorted([name for name in _MODELS_JSON.keys()])

def get_brainchop_cmd(
    path, model: str|None=None, args: list[str]=[], output_dir:Path|str|None=None) -> tuple[list[str], Path]:
  model_strs = ["-m", model] if model else []
  cmd = ["brainchop"]  + args + model_strs + [str(path)]
  cmd_hash = hashlib.md5(" ".join(cmd).encode()).hexdigest()[:8]
  output_path = CACHEDIR / str(cmd_hash+".nii.gz") if not output_dir else Path(output_dir)
  return cmd + ["-o", str(output_path)], output_path

def get_mrpeek_cmd(path) -> list[str]:
  return ["mrpeek"] + ["-batch"] + [str(path)]

def cmd_to_str(l: list[str]) -> str:
  return " ".join(l)


# list available models
print("available models:", _MODELS)

# 0. download files
test_files = ["t1_crop"]
paths = [fetch(_URLS[name], name + ".nii.gz") for name in test_files]

# 1. print mrpeek commands for original files
print("="*80)
print("mrpeek commands for original files:")
mrpeek_cmds = [get_mrpeek_cmd(path) for path in paths]
for cmd in mrpeek_cmds: print(cmd_to_str(cmd))

# 2. print brainchop commands (no args) # paths are deterministic given model, input filename and args
print("="*80)
print("brainchop commands:")
brainchop_cmds = [get_brainchop_cmd(path, model)[0] for path in paths for model in _MODELS]
output_paths   = [get_brainchop_cmd(path, model)[1] for path in paths for model in _MODELS]
for cmd in brainchop_cmds: print(cmd_to_str(cmd))
for path in output_paths: path.parent.mkdir(parents=True, exist_ok=True)

# 3. print mrpeek commnd for output files
output_mrpeek_cmds = mrpeek_cmds = [get_mrpeek_cmd(path) for path in output_paths]
for cmd in output_mrpeek_cmds: print(cmd_to_str(cmd))

if getenv("DRYRUN"): exit(0)

# 4. run all commands (sequential for now)
all_cmds = zip(mrpeek_cmds, brainchop_cmds, output_mrpeek_cmds)
for cmd_pack in all_cmds: 
  for cmd in cmd_pack:
    print(">>> RUNNING: ", cmd_to_str(cmd))
    subprocess.run(cmd)
  print("=" * 80)
