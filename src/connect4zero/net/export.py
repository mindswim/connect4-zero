"""
Export model to ONNX format for browser inference.
"""

from __future__ import annotations

import torch
import torch.onnx
from pathlib import Path

from .model import Connect4Net, load_checkpoint
from ..game import NUM_CHANNELS, ROWS, COLS


def export_to_onnx(
    checkpoint_path: str,
    output_path: str,
    opset_version: int = 14,
) -> None:
    """
    Export a trained model to ONNX format.

    Args:
        checkpoint_path: Path to .pt checkpoint
        output_path: Path for output .onnx file
        opset_version: ONNX opset version (14 works well with onnxruntime-web)
    """
    # Load model
    model, _ = load_checkpoint(checkpoint_path, device=torch.device("cpu"))
    model.eval()

    # Create dummy input
    dummy_input = torch.randn(1, NUM_CHANNELS, ROWS, COLS)

    # Export
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["board"],
        output_names=["policy", "value"],
        dynamic_axes={
            "board": {0: "batch_size"},
            "policy": {0: "batch_size"},
            "value": {0: "batch_size"},
        },
    )

    print(f"Exported to {output_path}")

    # Verify the export
    try:
        import onnx
        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model verified successfully")
    except ImportError:
        print("Install 'onnx' package to verify: pip install onnx")


def export_for_web(
    checkpoint_path: str,
    output_dir: str = "web/public",
) -> None:
    """
    Export model optimized for web deployment.

    Creates:
    - model.onnx: The ONNX model
    - model_config.json: Model metadata for the frontend
    """
    import json

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Export ONNX
    onnx_path = output_dir / "model.onnx"
    export_to_onnx(checkpoint_path, str(onnx_path))

    # Load model to get config
    model, checkpoint = load_checkpoint(checkpoint_path, device=torch.device("cpu"))

    # Write config
    config = {
        "num_channels": model.num_channels,
        "num_blocks": model.num_blocks,
        "board_rows": ROWS,
        "board_cols": COLS,
        "input_channels": NUM_CHANNELS,
    }

    config_path = output_dir / "model_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"Config written to {config_path}")
