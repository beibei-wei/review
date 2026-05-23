import os

import torch
import time
from Networks.net import MODEL as net
from thop import profile  # 加入 GFLOPs 计算


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda:0')


# model_path = "./models/model.pth"

input_h = 480
input_w = 640
warmup = 10
test_times = 100


def load_model():
    model = net(in_channel=1)
    use_gpu = torch.cuda.is_available()

    if use_gpu:
        model = model.cuda()
    #  else:
    #      state_dict = torch.load(model_path, map_location='cpu')
    #      model.load_state_dict(state_dict)

    model.eval()
    return model


def count_params(model):
    total = sum(p.numel() for p in model.parameters())
    print(f"Params: {total / 1e6:.2f} M")
    return total


def count_gflops(model):
    ir = torch.randn(1, 1, input_h, input_w).to(device)
    vi = torch.randn(1, 1, input_h, input_w).to(device)

    with torch.no_grad():
        flops, _ = profile(model, inputs=(ir, vi), verbose=False)

    total_gflops = flops / 1e9
    print(f" GFLOPs：{total_gflops:.2f} G")
    return total_gflops


def test_speed(model):
    print(f"warmup {warmup}  ...")
    with torch.no_grad():
        for _ in range(warmup):
            ir = torch.randn(1, 1, input_h, input_w).to(device)
            vi = torch.randn(1, 1, input_h, input_w).to(device)
            model(ir, vi)

    print(f"test times: {test_times}  ...")
    total_time = 0.0
    with torch.no_grad():
        for _ in range(test_times):
            ir = torch.randn(1, 1, input_h, input_w).to(device)
            vi = torch.randn(1, 1, input_h, input_w).to(device)

            torch.cuda.synchronize()
            t0 = time.time()

            model(ir, vi)

            torch.cuda.synchronize()
            t1 = time.time()
            total_time += t1 - t0

    avg_ms = total_time / test_times * 1000
    fps = 1000 / avg_ms
    return avg_ms, fps


if __name__ == '__main__':
    model = load_model()

    print("\n" + "=" * 50)
    count_params(model)
    count_gflops(model)
    print("=" * 50)

    avg_time, fps = test_speed(model)

    print("\n" + "=" * 50)
    print(f"Avg Inference Time: {avg_time:.2f} ms")
    print(f" FPS: {fps:.2f}")
    print("=" * 50)