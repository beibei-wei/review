import os
import torch
import time
import numpy as np
import torch.nn as nn
from sleepnet import DE_Encoder, DE_Decoder, LowFreqExtractor, HighFreqExtractor
from thop import profile

# ===================== =====================
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

input_h = 480
input_w = 640
warmup = 10
test_times = 100


# =================================================================

def load_models():

    Encoder = DE_Encoder().to(device)
    Decoder = DE_Decoder().to(device)
    LFExtractor = LowFreqExtractor(dim=64).to(device)
    HFExtractor = HighFreqExtractor(num_layers=3).to(device)

    Encoder.eval()
    Decoder.eval()
    LFExtractor.eval()
    HFExtractor.eval()

    return Encoder, Decoder, LFExtractor, HFExtractor



def count_total_params(Encoder, Decoder, LFExtractor, HFExtractor):
    p1 = sum(p.numel() for p in Encoder.parameters())
    p2 = sum(p.numel() for p in Decoder.parameters())
    p3 = sum(p.numel() for p in LFExtractor.parameters())
    p4 = sum(p.numel() for p in HFExtractor.parameters())
    total = p1 + p2 + p3 + p4

    print(f"Params：{total / 1e6:.2f} M")
    return total


# ==========  GFLOPs ==========
def count_gflops(Encoder, Decoder, LFExtractor, HFExtractor):
    vi = torch.randn(1, 1, input_h, input_w).to(device)
    ir = torch.randn(1, 1, input_h, input_w).to(device)
    dummy_base = vi * 0.5 + ir * 0.5

    total_flops = 0

    with torch.no_grad():
        flops_enc_vi, _ = profile(Encoder, inputs=(vi,), verbose=False)
        flops_enc_ir, _ = profile(Encoder, inputs=(ir,), verbose=False)
        total_flops += flops_enc_vi + flops_enc_ir

        fv_b, fv_d, fv = Encoder(vi)
        fi_b, fi_d, fi = Encoder(ir)

        flops_lf, _ = profile(LFExtractor, inputs=(fi_b + fv_b,), verbose=False)
        total_flops += flops_lf

        flops_hf, _ = profile(HFExtractor, inputs=(fi_d + fv_d,), verbose=False)
        total_flops += flops_hf

        ff_b = LFExtractor(fi_b + fv_b)
        ff_d = HFExtractor(fi_d + fv_d)
        flops_dec, _ = profile(Decoder, inputs=(dummy_base, ff_b, ff_d), verbose=False)
        total_flops += flops_dec

    total_gflops = total_flops / 1e9
    print(f"  GFLOPs：{total_gflops:.2f} G")
    return total_gflops



def test_speed(Encoder, Decoder, LFExtractor, HFExtractor):
    print(f"warmup {warmup}  ...")
    with torch.no_grad():
        for _ in range(warmup):
            vi = torch.randn(1, 1, input_h, input_w).to(device)
            ir = torch.randn(1, 1, input_h, input_w).to(device)

            fv_b, fv_d, fv = Encoder(vi)
            fi_b, fi_d, fi = Encoder(ir)
            ff_b = LFExtractor(fi_b + fv_b)
            ff_d = HFExtractor(fi_d + fv_d)
            fuse, _ = Decoder(vi * 0.5 + ir * 0.5, ff_b, ff_d)

    print(f"test times {test_times}  ...")
    total_time = 0.0
    with torch.no_grad():
        for _ in range(test_times):
            vi = torch.randn(1, 1, input_h, input_w).to(device)
            ir = torch.randn(1, 1, input_h, input_w).to(device)

            torch.cuda.synchronize()
            t0 = time.time()

            fv_b, fv_d, fv = Encoder(vi)
            fi_b, fi_d, fi = Encoder(ir)
            ff_b = LFExtractor(fi_b + fv_b)
            ff_d = HFExtractor(fi_d + fv_d)
            fuse, _ = Decoder(vi * 0.5 + ir * 0.5, ff_b, ff_d)

            torch.cuda.synchronize()
            t1 = time.time()
            total_time += t1 - t0

    avg_ms = total_time / test_times * 1000
    fps = 1000 / avg_ms
    return avg_ms, fps



if __name__ == '__main__':
    models = load_models()

    print("\n" + "=" * 60)
    count_total_params(*models)
    count_gflops(*models)
    print("=" * 60)

    avg_time, fps = test_speed(*models)

    print("\n" + "=" * 60)
    print(f"Avg Inference Time: {avg_time:.2f} ms")
    print(f" FPS：{fps:.2f}")
    print("=" * 60)