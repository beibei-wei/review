import os
import numpy as np
import torch
import torch.nn as nn
from thop import profile, clever_format
import time
import warnings
warnings.filterwarnings("ignore")



device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")


input_h = 480
input_w = 640
warmup = 10
test_times = 50
ckpt_path = "models/MKDFusionv1.pth"     ## set weights path


def load_model():
    from MKDFusionv1 import (
        BaseFeatureExtraction,
        DetailFeatureExtraction,
        VMamba_Encoder,
        VMamba_Decoder,
        multihead,
        multihead2
    )


    Encoder = VMamba_Encoder().to(device)
    Decoder = VMamba_Decoder().to(device)
    BaseFuse = BaseFeatureExtraction(dim=64, num_heads=8).to(device)
    DetailFuse = DetailFeatureExtraction(num_layers=1).to(device)
    MultiHead = multihead().to(device)
    MultiHead2 = multihead2().to(device)


    state = torch.load(ckpt_path, map_location=device)
    Encoder.load_state_dict({k.replace("module.", ""): v for k, v in state['VMamba_Encoder'].items()})
    Decoder.load_state_dict({k.replace("module.", ""): v for k, v in state['VMamba_Decoder'].items()})
    BaseFuse.load_state_dict({k.replace("module.", ""): v for k, v in state['BaseFuseLayer'].items()})
    DetailFuse.load_state_dict({k.replace("module.", ""): v for k, v in state['DetailFuseLayer'].items()})
    MultiHead.load_state_dict({k.replace("module.", ""): v for k, v in state['multihead'].items()})
    MultiHead2.load_state_dict({k.replace("module.", ""): v for k, v in state['multihead2'].items()})


    Encoder.eval()
    Decoder.eval()
    BaseFuse.eval()
    DetailFuse.eval()
    MultiHead.eval()
    MultiHead2.eval()

    return Encoder, Decoder, BaseFuse, DetailFuse, MultiHead, MultiHead2


def get_total_params(models):
    total = 0
    for m in models:
        total += sum(p.numel() for p in m.parameters())
    return total

def compute_flops(Encoder, Decoder, BaseFuse, DetailFuse, MultiHead, MultiHead2):

    ir = torch.randn(1, 1, input_h, input_w).to(device)
    vi = torch.randn(1, 1, input_h, input_w).to(device)

    total_flops = 0

    with torch.no_grad():

        flops_enc_v, _ = profile(Encoder, inputs=(vi,), verbose=False)
        flops_enc_i, _ = profile(Encoder, inputs=(ir,), verbose=False)
        total_flops += flops_enc_v + flops_enc_i

        fv_b, fv_d, fv = Encoder(vi)
        fi_b, fi_d, fi = Encoder(ir)


        flops_mh, _ = profile(MultiHead, inputs=(fv_b, fi_b), verbose=False)
        total_flops += flops_mh
        flops_mh2, _ = profile(MultiHead2, inputs=(fv_d, fi_d), verbose=False)
        total_flops += flops_mh2

        fvi_b = MultiHead(fv_b, fi_b)
        fvi_d = MultiHead2(fv_d, fi_d)


        flops_base, _ = profile(BaseFuse, inputs=(fvi_b,), verbose=False)
        total_flops += flops_base
        flops_det, _ = profile(DetailFuse, inputs=(fvi_d,), verbose=False)
        total_flops += flops_det

        ff_b = BaseFuse(fvi_b)
        ff_d = DetailFuse(fvi_d)


        flops_dec, _ = profile(Decoder, inputs=(vi, ff_b, ff_d), verbose=False)
        total_flops += flops_dec

    return total_flops

def compute_speed(Encoder, Decoder, BaseFuse, DetailFuse, MultiHead, MultiHead2):
    ir = torch.randn(1, 1, input_h, input_w).to(device)
    vi = torch.randn(1, 1, input_h, input_w).to(device)


    print(f"warmup {warmup}  ...")
    with torch.no_grad():
        for _ in range(warmup):
            fv_b, fv_d, _ = Encoder(vi)
            fi_b, fi_d, _ = Encoder(ir)
            fvi_b = MultiHead(fv_b, fi_b)
            fvi_d = MultiHead2(fv_d, fi_d)
            ff_b = BaseFuse(fvi_b)
            ff_d = DetailFuse(fvi_d)
            _, _ = Decoder(vi, ff_b, ff_d)


    print(f"test times: {test_times}  ...")
    total_time = 0.0
    with torch.no_grad():
        for _ in range(test_times):
            torch.cuda.synchronize()
            t0 = time.time()


            fv_b, fv_d, _ = Encoder(vi)
            fi_b, fi_d, _ = Encoder(ir)
            fvi_b = MultiHead(fv_b, fi_b)
            fvi_d = MultiHead2(fv_d, fi_d)
            ff_b = BaseFuse(fvi_b)
            ff_d = DetailFuse(fvi_d)
            data_Fuse, _ = Decoder(vi, ff_b, ff_d)

            torch.cuda.synchronize()
            t1 = time.time()
            total_time += t1 - t0

    avg_time = total_time / test_times * 1000
    fps = 1000.0 / avg_time
    return avg_time, fps

if __name__ == '__main__':
    models = load_model()
    total_params = get_total_params(models)
    total_flops = compute_flops(*models)
    avg_time, fps = compute_speed(*models)


    flops_fmt, params_fmt = clever_format([total_flops, total_params], "%.3f")

    print("\n" + "=" * 70)
    print(f" Params: {total_params / 1e6:.2f} M  ({params_fmt})")
    print(f" GFLOPs: {total_flops / 1e9:.2f} G  ({flops_fmt})")
    print(f" Avg Inference Time: {avg_time:.2f} ms")
    print(f" FPS: {fps:.2f}")
    print("=" * 70)