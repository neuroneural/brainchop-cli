import requests
from pathlib import Path

AVAILABLE_MODELS = {
  "small": "model18cls",
  "medium": "model20chan3cls",
  "large": "model30chan18cls",
  "astronomical": "model30chan50cls",
  # Add more models here as they become available
}

BASE_URL = "https://github.com/neuroneural/brainchop/raw/master/public/models/"

def list_available_models():
  print("Available models:")
  for model in AVAILABLE_MODELS:
    print(f"- {model}")

def download_model(model_name):
  if model_name not in AVAILABLE_MODELS:
    print(f"Error: Model '{model_name}' is not available.")
    return None

  model_dir = AVAILABLE_MODELS[model_name]
  cache_dir = Path.home() / ".cache" / "brainchop" / "models" / model_dir
  cache_dir.mkdir(parents=True, exist_ok=True)

  files_to_download = ["model.json", "model.bin"]
  downloaded_paths = {}

  for file in files_to_download:
    url = f"{BASE_URL}{model_dir}/{file}"
    local_path = cache_dir / file
    
    if not local_path.exists():
      print(f"Downloading {file} for model {model_name}...")
      response = requests.get(url)
      if response.status_code == 200:
        with open(local_path, "wb") as f:
          f.write(response.content)
        print(f"Downloaded {file} to {local_path}")
      else:
        print(f"Failed to download {file}. Status code: {response.status_code}")
        return None
    else:
      print(f"{file} already exists at {local_path}")
    
    downloaded_paths[file] = str(local_path)

  return downloaded_paths

if __name__ == "__main__":
    # Test the functions
    list_available_models()
    # result = download_model("gwm_30")
    # print(result)
