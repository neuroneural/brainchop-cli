"""
Generate models.rst documentation from models.json.

This script reads the models.json file and generates comprehensive
documentation for all available models.
"""

import json
from pathlib import Path


def generate_models_rst():
    """Generate models.rst from the models.json file."""
    
    # Path to models.json in repo root
    models_json_path = Path(__file__).parent.parent / "models.json"
    
    if not models_json_path.exists():
        print(f"Warning: {models_json_path} not found, skipping model documentation generation")
        return
    
    # Load models data
    with open(models_json_path, 'r') as f:
        models_data = json.load(f)
    
    # Build RST content
    lines = [
        "Available Models",
        "================",
        "",
        "BrainChop supports multiple pre-trained segmentation models, each optimized for different tasks.",
        "",
        "Model Overview",
        "--------------",
        "",
    ]
    
    # Add summary table
    lines.extend([
        ".. list-table:: Model Comparison",
        "   :header-rows: 1",
        "   :widths: 20 50 30",
        "",
        "   * - Model Name",
        "     - Description", 
        "     - Normalization",
    ])
    
    for model_name, model_info in models_data.items():
        description = model_info.get('description', 'No description available')
        normalization = model_info.get('considerations', 'Unknown')
        
        lines.append(f"   * - ``{model_name}``")
        lines.append(f"     - {description}")
        lines.append(f"     - {normalization}")
    
    lines.extend([
        "",
        "Detailed Model Information",
        "--------------------------",
        "",
    ])
    
    # Add detailed sections for each model
    for model_name, model_info in models_data.items():
        # Model header
        lines.extend([
            model_name,
            "~" * len(model_name),
            "",
        ])
        
        # Description
        description = model_info.get('description', 'No description available')
        lines.append(description)
        lines.append("")
        
        # Details list
        lines.append("**Details:**")
        lines.append("")
        
        folder = model_info.get('folder', 'N/A')
        normalization = model_info.get('considerations', 'Unknown')
        parameter = model_info.get('parameter_name', model_name)
        
        lines.extend([
            f"* **Model Folder:** ``{folder}``",
            f"* **Normalization:** {normalization}",
            f"* **CLI Parameter:** ``{parameter}``",
            "",
        ])
        
        # Usage example
        lines.extend([
            "**Usage Example:**",
            "",
            ".. code-block:: bash",
            "",
            f"   brainchop input.nii.gz -m {model_name} -o output.nii.gz",
            "",
        ])
    
    # Add model sources section
    lines.extend([
        "Model Sources",
        "-------------",
        "",
        "All models are automatically downloaded from the BrainChop model repository:",
        "",
        "* **GitHub Repository:** https://github.com/neuroneural/brainchop-models",
        "* **Base URL:** https://github.com/neuroneural/brainchop-models/raw/main/meshnet/",
        "",
        "Models are cached locally in ``~/.cache/brainchop/models/`` after first download.",
        "",
        "Updating Models",
        "~~~~~~~~~~~~~~~",
        "",
        "To update the model listing:",
        "",
        ".. code-block:: bash",
        "",
        "   brainchop --update",
        "",
        "Model Architecture Formats",
        "---------------------------",
        "",
        "BrainChop supports two model architecture formats:",
        "",
        "**New Architecture Format (.pth weights)**",
        "",
        "* Uses PyTorch-style weight format",
        "* Modern JSON-based architecture description",
        "* Better performance and flexibility",
        "* Recommended for new models",
        "",
        "**Legacy Architecture Format (.bin weights)**", 
        "",
        "* Uses TensorFlow.js weight format",
        "* Legacy JSON architecture description",
        "* Maintained for backward compatibility",
        "",
        "The architecture format is automatically detected based on the model configuration.",
        "",
    ])
    
    # Write to file
    output_path = Path(__file__).parent / "models.rst"
    with open(output_path, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f"Generated {output_path}")


if __name__ == "__main__":
    generate_models_rst()