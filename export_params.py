import argparse
import json
import torch
from pathlib import Path
from rsic import get_model

def main():
    parser = argparse.ArgumentParser(description="Export entropy parameters for board C++ deployment")
    parser.add_argument("--checkpoint", type=str, default="best.pt", help="Path to the trained model checkpoint")
    parser.add_argument("--output", type=str, default="params.json", help="Path to save the parameters JSON file")
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}...")
    raw = torch.load(args.checkpoint, map_location="cpu")
    decoder_type = raw.get("decoder_type", "swin")
    
    model = get_model(decoder_type=decoder_type)
    model.load_state_dict(raw.get("state_dict", raw), strict=False)
    model.eval()

    z_entropy = model.entropy_bottleneck_z
    
    # 获取 z_medians，如果没有这个属性则默认填充 0
    if hasattr(z_entropy, 'medians'):
        z_medians = z_entropy.medians.detach().cpu().to(torch.float32).tolist()
    else:
        z_medians = [0.0] * int(model.Z)
    
    payload = {
        "format": "compressai-nano-hyper-entropy-params-v1",
        "model_variant": model.model_variant,
        "cnz4_supported": False,
        "note": "OpenRSIC exported entropy parameters for RK3588 board deployment",
        "channels_y": int(model.M),
        "channels_z": int(model.Z),
        "model_type": model.config.model_type,
        "has_means_y": bool(model.config.model_type == "mean_scale_hyperprior"),
        "quant_step_y": float(model.conditional_entropy_y.quant_step.detach().cpu()),
        "quant_step_z": float(z_entropy.quant_step.detach().cpu()),
        "downsampling_factor": int(model.downsampling_factor),
        "model_config_name": model.config.name,
        "scale_min": float(model.scale_min),
        "scale_max": float(model.scale_max),
        "z_medians": z_medians,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    
    print(f"Successfully exported entropy parameters to {args.output}")
    print(f" - channels_y: {payload['channels_y']}")
    print(f" - channels_z: {payload['channels_z']}")
    print(f" - quant_step_y: {payload['quant_step_y']}")
    print(f" - quant_step_z: {payload['quant_step_z']}")

if __name__ == "__main__":
    main()
