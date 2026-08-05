import torch
import torch.nn as nn
from pathlib import Path
from rsic import get_model, QATSettings

# 新增一个包装类，用于调用 analysis_transform
class AnalysisWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        
    def forward(self, x):
        # 依次返回 y(latent), z(hyper_latent), scales_y, means_y
        return self.model.analysis_transform(x)

def load_checkpoint_and_export_onnx(ckpt_path: str, onnx_path: str, img_size: int = 512):
    print(f"Loading checkpoint from {ckpt_path}...")
    raw = torch.load(ckpt_path, map_location="cpu")
    
    decoder_type = raw.get("decoder_type", "swin")
    quality_profile = raw.get("quality_profile", "rsic_fp")
    
    # QAT的配置恢复
    if quality_profile == "rsic_qat8":
        qat = QATSettings(
            enable_latent_fake_quant=True, latent_fake_quant_bits=8, latent_fake_quant_clip=6.0,
            enable_z_fake_quant=True, z_fake_quant_bits=8, z_fake_quant_clip=6.0,
            enable_scale_fake_quant=True, scale_fake_quant_bits=8, scale_fake_quant_clip=8.0,
        )
    else:
        qat = QATSettings()
        
    model = get_model(decoder_type=decoder_type, qat=qat)
    model.load_state_dict(raw.get("state_dict", raw), strict=False)
    model.eval()
    
    # 【修改这里】：使用 Wrapper 包装整个分析变换过程
    wrapper = AnalysisWrapper(model)
    wrapper.eval()
    
    print(f"Exporting analysis transform to {onnx_path} with resolution {img_size}x{img_size}...")
    dummy_input = torch.randn(1, 3, img_size, img_size)
    
    # 导出为ONNX，指定 4 个输出名称，严格对应板端 C++ 所需
    torch.onnx.export(
        wrapper,
        dummy_input,
        onnx_path,
        input_names=["input"],
        output_names=["latent", "hyper_latent", "scales_y", "means_y"],
        opset_version=13,
        do_constant_folding=True,
    )
    print(f"Successfully exported ONNX model to {onnx_path}")

if __name__ == "__main__":
    load_checkpoint_and_export_onnx("./OpenRSIC-main/best.pt", "./OpenRSIC-main/encoder_512x512.onnx", img_size=512)
