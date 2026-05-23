import torch
import time
import torch.nn.functional as F
from S4Fusion import *
from thop import profile


device = "cuda:0"
input_h = 480
input_w = 640
warmup = 10
test_times = 100


def get_rest(x):
    add = 0
    while (x - 1) % 3 != 0 or ((x - 1) // 3) % 4 != 0:
        add += 1
        x += 1
    return add


def compute_pad(x):
    rest = get_rest(x)
    left = rest // 2
    left_pad = left
    right_pad = rest - left
    new_x = left_pad + x + right_pad
    return left_pad, right_pad, new_x


left1_pad, right1_pad, new_h = compute_pad(input_h)
left2_pad, right2_pad, new_w = compute_pad(input_w)


def load_model():
    model = MambaNet().to(device)

    model.eval()
    return model


def count_params(model, name="MambaNet"):
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total / 1e6:.2f} M")
    return total


def count_gflops(model):
    dummy_ir = torch.randn(1, 1, new_h, new_w).to(device)
    dummy_vi = torch.randn(1, 1, new_h, new_w).to(device)

    with torch.no_grad():
        flops, _ = profile(model, inputs=(dummy_ir, dummy_vi), verbose=False)

    gflops = flops / 1e9
    print(f" GFLOPs：{gflops:.2f} G")
    return gflops


def test_inference_speed(model):
    print(f"warmup {warmup}  ...")
    with torch.no_grad():
        for _ in range(warmup):
            dummy_ir = torch.randn(1, 1, new_h, new_w).to(device)
            dummy_vi = torch.randn(1, 1, new_h, new_w).to(device)
            model(dummy_ir, dummy_vi)

    print(f"test times: {test_times} ...")
    total_time = 0.0
    with torch.no_grad():
        for _ in range(test_times):
            dummy_ir = torch.randn(1, 1, new_h, new_w).to(device)
            dummy_vi = torch.randn(1, 1, new_h, new_w).to(device)

            torch.cuda.synchronize(device)
            start = time.time()
            model(dummy_ir, dummy_vi)
            torch.cuda.synchronize(device)
            end = time.time()
            total_time += (end - start)

    avg_time = total_time / test_times * 1000
    fps = 1000 / avg_time
    return avg_time, fps


if __name__ == '__main__':
    model = load_model()

    print("\n" + "=" * 50)
    count_params(model)
    count_gflops(model)
    print("=" * 50)

    avg_time, fps = test_inference_speed(model)

    print("\n" + "=" * 50)
    print(f"  Avg Inference Time: {avg_time:.2f} ms")
    print(f"  FPS: {fps:.2f}")
    print("=" * 50)