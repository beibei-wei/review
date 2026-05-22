import os
import numpy as np
import cv2
import torch
from torchvision.transforms import functional as F
from PIL import Image
from model.CAWM_Mamba import WaveMamba
from thop import profile, clever_format
import time
import warnings

warnings.filterwarnings('ignore')


os.environ["CUDA_VISIBLE_DEVICES"] = "1"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_h = 480
input_w = 640
warmup = 10
test_times = 50
weights_path = "./Single.pth"   ## set weights path



def load_model():

    model = WaveMamba().to(device)

    state_dict = torch.load(weights_path, map_location=device)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(new_state_dict)

    model.eval()
    return model



def compute_flops_params(model):
    ir = torch.randn(1, 3, input_h, input_w).to(device)
    vi = torch.randn(1, 3, input_h, input_w).to(device)


    flops, params = profile(model, inputs=(ir, vi), verbose=False)
    flops_fmt, params_fmt = clever_format([flops, params], "%.3f")

    return flops, params, flops_fmt, params_fmt



def compute_speed(model):
    ir = torch.randn(1, 3, input_h, input_w).to(device)
    vi = torch.randn(1, 3, input_h, input_w).to(device)

    print(f"warmup {warmup} ...")
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(ir, vi)

    print(f"test time {test_times}...")
    total_time = 0.0
    with torch.no_grad():
        for _ in range(test_times):
            torch.cuda.synchronize()
            t0 = time.time()

            _ = model(ir, vi)

            torch.cuda.synchronize()
            t1 = time.time()
            total_time += t1 - t0

    avg_time = total_time / test_times * 1000
    fps = 1000.0 / avg_time
    return avg_time, fps



if __name__ == '__main__':
    model = load_model()

    flops, params, flops_fmt, params_fmt = compute_flops_params(model)
    avg_time, fps = compute_speed(model)

    print("\n" + "=" * 65)
    print(f" Params: {params / 1e6:.2f} M ")
    print(f" GFLOPs: {flops / 1e9:.2f} G ")
    print(f" Avg Inference Time {avg_time:.2f} ms")
    print(f" FPS: {fps:.2f}")
    print("=" * 65)