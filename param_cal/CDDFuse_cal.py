import os
import torch
import time
import numpy as np
import torch.nn as nn
from net import Restormer_Encoder, Restormer_Decoder, BaseFeatureExtraction, DetailFeatureExtraction
from thop import profile


os.environ["CUDA_VISIBLE_DEVICES"] = "0"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ckpt_path = "models/CDDFuse_IVF.pth"       ## set weights path

input_h = 480
input_w = 640
warmup = 10
test_times = 100


def load_models():

    Encoder = Restormer_Encoder().to(device)
    Decoder = Restormer_Decoder().to(device)
    BaseFuseLayer = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
    DetailFuseLayer = DetailFeatureExtraction(num_layers=1).to(device)


    state = torch.load(ckpt_path, map_location=device)
    Encoder.load_state_dict({k.replace("module.", ""): v for k, v in state['DIDF_Encoder'].items()})
    Decoder.load_state_dict({k.replace("module.", ""): v for k, v in state['DIDF_Decoder'].items()})
    BaseFuseLayer.load_state_dict({k.replace("module.", ""): v for k, v in state['BaseFuseLayer'].items()})
    DetailFuseLayer.load_state_dict({k.replace("module.", ""): v for k, v in state['DetailFuseLayer'].items()})

    Encoder.eval()
    Decoder.eval()
    BaseFuseLayer.eval()
    DetailFuseLayer.eval()

    return Encoder, Decoder, BaseFuseLayer, DetailFuseLayer

# ==========      ==========
def count_total_params(Encoder, Decoder, BaseFuseLayer, DetailFuseLayer):
    p_enc = sum(p.numel() for p in Encoder.parameters())
    p_dec = sum(p.numel() for p in Decoder.parameters())
    p_base = sum(p.numel() for p in BaseFuseLayer.parameters())
    p_detail = sum(p.numel() for p in DetailFuseLayer.parameters())
    total = p_enc + p_dec + p_base + p_detail
    print(f" Params: {total / 1e6:.2f} M  ")
    return total

# ==========   GFLOPs ==========
def count_gflops(Encoder, Decoder, BaseFuseLayer, DetailFuseLayer):
    vi = torch.randn(1, 1, input_h, input_w).to(device)
    ir = torch.randn(1, 1, input_h, input_w).to(device)

    total_flops = 0

    with torch.no_grad():
        flops_enc_vi, _ = profile(Encoder, inputs=(vi,), verbose=False)
        flops_enc_ir, _ = profile(Encoder, inputs=(ir,), verbose=False)
        total_flops += flops_enc_vi + flops_enc_ir

        fv_b, fv_d, _ = Encoder(vi)
        fi_b, fi_d, _ = Encoder(ir)

        flops_base, _ = profile(BaseFuseLayer, inputs=(fv_b + fi_b,), verbose=False)
        total_flops += flops_base

        flops_detail, _ = profile(DetailFuseLayer, inputs=(fv_d + fi_d,), verbose=False)
        total_flops += flops_detail

        ff_b = BaseFuseLayer(fv_b + fi_b)
        ff_d = DetailFuseLayer(fv_d + fi_d)
        flops_dec, _ = profile(Decoder, inputs=(vi, ff_b, ff_d), verbose=False)
        total_flops += flops_dec

    total_gflops = total_flops / 1e9
    print(f" GFLOPs: {total_gflops:.2f} G  ")
    return total_gflops

# ==========        ==========
def test_speed(Encoder, Decoder, BaseFuseLayer, DetailFuseLayer):
    print(f" warmup {warmup} ...")
    with torch.no_grad():
        for _ in range(warmup):
            vi = torch.randn(1, 1, input_h, input_w).to(device)
            ir = torch.randn(1, 1, input_h, input_w).to(device)

            fv_b, fv_d, _ = Encoder(vi)
            fi_b, fi_d, _ = Encoder(ir)
            ff_b = BaseFuseLayer(fv_b + fi_b)
            ff_d = DetailFuseLayer(fv_d + fi_d)
            fuse, _ = Decoder(vi, ff_b, ff_d)

    print(f" test times {test_times} ...")
    total_time = 0.0
    with torch.no_grad():
        for _ in range(test_times):
            vi = torch.randn(1, 1, input_h, input_w).to(device)
            ir = torch.randn(1, 1, input_h, input_w).to(device)

            torch.cuda.synchronize()
            t0 = time.time()

            fv_b, fv_d, _ = Encoder(vi)
            fi_b, fi_d, _ = Encoder(ir)
            ff_b = BaseFuseLayer(fv_b + fi_b)
            ff_d = DetailFuseLayer(fv_d + fi_d)
            fuse, _ = Decoder(vi, ff_b, ff_d)

            torch.cuda.synchronize()
            t1 = time.time()
            total_time += t1 - t0

    avg_ms = total_time / test_times * 1000
    fps = 1000 / avg_ms
    return avg_ms, fps


if __name__ == '__main__':
    models = load_models()

    print("\n" + "="*65)
    count_total_params(*models)
    count_gflops(*models)
    print("="*65)

    avg_time, fps = test_speed(*models)


    print(f" Avg Inference Time {avg_time:.2f} ms")
    print(f" FPS: {fps:.2f}")
    print("=" * 65)