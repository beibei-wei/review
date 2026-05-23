import os
import time
import torch
import numpy as np
from thop import profile
from network_WMamba import WMamba as net
from cut_recon import crop_image_overlap, recon_image_overlap


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = net(in_chans=1, embed_dim=192).to(device)
model.eval()


dummy_ir = torch.randn(1, 1, 480, 640).to(device)
dummy_vi = torch.randn(1, 1, 480, 640).to(device)
flops, params = profile(model, inputs=(dummy_ir, dummy_vi), verbose=False)

crop_size = 128
test_ir = torch.randn(1, 1, 480, 640).to(device)
test_vi = torch.randn(1, 1, 480, 640).to(device)


with torch.no_grad():
    ir_crop, h_pad, w_pad = crop_image_overlap(test_ir, (crop_size, crop_size))
    vi_crop, _, _ = crop_image_overlap(test_vi, (crop_size, crop_size))
    out_crop = torch.zeros_like(ir_crop)
    for ii in range(ir_crop.shape[0]):
        out_crop[ii:ii+1] = model(ir_crop[ii:ii+1], vi_crop[ii:ii+1])
    _ = recon_image_overlap(out_crop, test_ir, (crop_size, crop_size), h_pad, w_pad)


repeat_times = 20
infer_times = []
for _ in range(repeat_times):
    torch.cuda.synchronize()
    start = time.time()
    with torch.no_grad():
        ir_crop, h_pad, w_pad = crop_image_overlap(test_ir, (crop_size, crop_size))
        vi_crop, _, _ = crop_image_overlap(test_vi, (crop_size, crop_size))
        out_crop = torch.zeros_like(ir_crop)
        for ii in range(ir_crop.shape[0]):
            out_crop[ii:ii+1] = model(ir_crop[ii:ii+1], vi_crop[ii:ii+1])
        _ = recon_image_overlap(out_crop, test_ir, (crop_size, crop_size), h_pad, w_pad)
    torch.cuda.synchronize()
    end = time.time()
    infer_times.append(end - start)

mean_time = np.mean(infer_times)
std_time = np.std(infer_times)
fps = 1.0 / mean_time


print("="*60)
print(f" Params:     {params / 1e6:.2f} M")
print(f" GFLOPs:      {flops / 1e9:.2f} G")
print(f"Avg Inference Time:      {mean_time:.4f} s ± {std_time:.4f} s")
print(f"FPS:                {fps:.2f}")
print("="*60)