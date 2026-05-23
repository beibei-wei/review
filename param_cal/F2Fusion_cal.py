import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import torch
import torch.nn.functional as F
from net import network_fusion
from contourlet import ContourDec, ContourRec
from loss_gradient import gradient_sobel64
from thop import profile, clever_format
import time


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# checkpoint_path = "./checkpoint/Epoch_15_iters_1200.model"

input_h = 480
input_w = 640
warmup = 10
test_times = 50


def load_model():
    # 只初始化，不加载任何权重
    model = network_fusion().to(device)
    model.eval()

    ct = ContourDec(nlevs=3).to(device)
    ict = ContourRec().to(device)
    gra_map64 = gradient_sobel64().to(device)
    return model, ct, ict, gra_map64


def compute_flops_params(model, ct, ict):
    vi = torch.randn(1, 1, input_h, input_w).to(device)
    ir = torch.randn(1, 1, input_h, input_w).to(device)

    m = input_h // 16
    n = input_w // 16
    vi = vi[:, :, :m * 16, :n * 16]
    ir = ir[:, :, :m * 16, :n * 16]

    with torch.no_grad():
        vi64_l, vi32_l, vi16_l, vi64_h, vi32_h, vi16_h = model.vr.vi_encoder(vi, ct)
        ir64_l, ir32_l, ir16_l, ir64_h, ir32_h, ir16_h = model.vr.ir_encoder(ir, ct)

    flops_enc, _ = profile(model.vr.vi_encoder, inputs=(vi, ct), verbose=False)
    flops_fuse, _ = profile(model.feature_fusion,
                            inputs=(vi64_l, ir64_l, vi32_l, ir32_l, vi16_l, ir16_l),
                            verbose=False)
    flops_dec, _ = profile(model.vr.decoder,
                           inputs=(vi64_l, vi32_l, vi16_l, vi64_h, vi32_h, vi16_h, ict),
                           verbose=False)

    total_flops = flops_enc * 2 + flops_fuse + flops_dec
    total_params = sum(p.numel() for p in model.parameters())

    flops_fmt, params_fmt = clever_format([total_flops, total_params], "%.3f")
    return total_flops, total_params, flops_fmt, params_fmt


def compute_speed(model, ct, ict, gra_map64):
    vi = torch.randn(1, 1, input_h, input_w).to(device)
    ir = torch.randn(1, 1, input_h, input_w).to(device)
    m = input_h // 16
    n = input_w // 16
    vi = vi[:, :, :m * 16, :n * 16]
    ir = ir[:, :, :m * 16, :n * 16]

    print(f"warmup {warmup} ...")
    with torch.no_grad():
        for _ in range(warmup):
            vi64_l, vi32_l, vi16_l, vi64_h, vi32_h, vi16_h = model.vr.vi_encoder(vi, ct)
            ir64_l, ir32_l, ir16_l, ir64_h, ir32_h, ir16_h = model.vr.ir_encoder(ir, ct)

            for k in range(8):
                com_vi = F.pad(torch.abs(gra_map64(vi64_h[k])), (1, 1, 1, 1), mode='replicate')
                com_ir = F.pad(torch.abs(gra_map64(ir64_h[k])), (1, 1, 1, 1), mode='replicate')
                vi64_h[k] = torch.ge(com_vi, com_ir).float() * vi64_h[k] + torch.ge(com_ir, com_vi).float() * ir64_h[k]
            for k in range(8):
                com_vi = F.pad(torch.abs(gra_map64(vi32_h[k])), (1, 1, 1, 1), mode='replicate')
                com_ir = F.pad(torch.abs(gra_map64(ir32_h[k])), (1, 1, 1, 1), mode='replicate')
                vi32_h[k] = torch.ge(com_vi, com_ir).float() * vi32_h[k] + torch.ge(com_ir, com_vi).float() * ir32_h[k]
            for k in range(8):
                com_vi = F.pad(torch.abs(gra_map64(vi16_h[k])), (1, 1, 1, 1), mode='replicate')
                com_ir = F.pad(torch.abs(gra_map64(ir16_h[k])), (1, 1, 1, 1), mode='replicate')
                vi16_h[k] = torch.ge(com_vi, com_ir).float() * vi16_h[k] + torch.ge(com_ir, com_vi).float() * ir16_h[k]

            f64_l, f32_l, f16_l, _, _, _, _, _, _, _, _, _ = model.feature_fusion(vi64_l, ir64_l, vi32_l, ir32_l,
                                                                                  vi16_l, ir16_l)
            out = model.vr.decoder(f64_l, f32_l, f16_l, vi64_h, vi32_h, vi16_h, ict)

    print(f"test times: {test_times}  ...")
    total_time = 0.0
    with torch.no_grad():
        for _ in range(test_times):
            torch.cuda.synchronize()
            t0 = time.time()

            vi64_l, vi32_l, vi16_l, vi64_h, vi32_h, vi16_h = model.vr.vi_encoder(vi, ct)
            ir64_l, ir32_l, ir16_l, ir64_h, ir32_h, ir16_h = model.vr.ir_encoder(ir, ct)

            for k in range(8):
                com_vi = F.pad(torch.abs(gra_map64(vi64_h[k])), (1, 1, 1, 1), mode='replicate')
                com_ir = F.pad(torch.abs(gra_map64(ir64_h[k])), (1, 1, 1, 1), mode='replicate')
                vi64_h[k] = torch.ge(com_vi, com_ir).float() * vi64_h[k] + torch.ge(com_ir, com_vi).float() * ir64_h[k]
            for k in range(8):
                com_vi = F.pad(torch.abs(gra_map64(vi32_h[k])), (1, 1, 1, 1), mode='replicate')
                com_ir = F.pad(torch.abs(gra_map64(ir32_h[k])), (1, 1, 1, 1), mode='replicate')
                vi32_h[k] = torch.ge(com_vi, com_ir).float() * vi32_h[k] + torch.ge(com_ir, com_vi).float() * ir32_h[k]
            for k in range(8):
                com_vi = F.pad(torch.abs(gra_map64(vi16_h[k])), (1, 1, 1, 1), mode='replicate')
                com_ir = F.pad(torch.abs(gra_map64(ir16_h[k])), (1, 1, 1, 1), mode='replicate')
                vi16_h[k] = torch.ge(com_vi, com_ir).float() * vi16_h[k] + torch.ge(com_ir, com_vi).float() * ir16_h[k]

            f64_l, f32_l, f16_l, _, _, _, _, _, _, _, _, _ = model.feature_fusion(vi64_l, ir64_l, vi32_l, ir32_l,
                                                                                  vi16_l, ir16_l)
            out = model.vr.decoder(f64_l, f32_l, f16_l, vi64_h, vi32_h, vi16_h, ict)

            torch.cuda.synchronize()
            t1 = time.time()
            total_time += t1 - t0

    avg_time = total_time / test_times * 1000
    fps = 1000.0 / avg_time
    return avg_time, fps


if __name__ == '__main__':
    model, ct, ict, gra_map64 = load_model()

    total_flops, total_params, flops_fmt, params_fmt = compute_flops_params(model, ct, ict)
    avg_time, fps = compute_speed(model, ct, ict, gra_map64)

    print("\n" + "=" * 60)
    print(f"Params: {total_params / 1e6:.2f} M  ({params_fmt})")
    print(f" GFLOPs: {total_flops / 1e9:.2f} G  ({flops_fmt})")
    print(f"Avg Inference Time: {avg_time:.2f} ms")
    print(f" FPS: {fps:.2f}")
    print("=" * 60)