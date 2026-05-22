import torch
import time
from nets.Ufuser import Ufuser
from thop import profile  # 计算 GFLOPs 需要这个库


device = "cuda:0" if torch.cuda.is_available() else "cpu"
path_model = "model/EMMA.pth"     ## set weights path


input_h = 480
input_w = 640

warmup = 10
test_times = 100


def load_model():
    model = Ufuser().to(device)
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.eval()
    return model



def count_params_and_flops(model, name="Ufuser (EMMA)"):

    dummy_vi = torch.randn(1, 1, input_h, input_w).to(device)
    dummy_ir = torch.randn(1, 1, input_h, input_w).to(device)


    total_params = sum(p.numel() for p in model.parameters())
    flops, _ = profile(model, inputs=(dummy_vi, dummy_ir), verbose=False)

    GFLOPs = flops / 1e9
    params_M = total_params / 1e6

    print(f" Params: {params_M:.2f} M")
    print(f" GFLOPs: {GFLOPs:.2f} G")
    return params_M, GFLOPs


def test_inference_speed(model):
    print(f"warmup {warmup}  ...")
    with torch.no_grad():
        for _ in range(warmup):
            dummy_ir = torch.randn(1, 1, input_h, input_w).to(device)
            dummy_vi = torch.randn(1, 1, input_h, input_w).to(device)
            model(dummy_vi, dummy_ir)

    print(f"test times: {test_times}  ...")
    total_time = 0.0

    with torch.no_grad():
        for _ in range(test_times):
            dummy_ir = torch.randn(1, 1, input_h, input_w).to(device)
            dummy_vi = torch.randn(1, 1, input_h, input_w).to(device)

            torch.cuda.synchronize()
            start = time.time()
            model(dummy_vi, dummy_ir)
            torch.cuda.synchronize()
            end = time.time()
            total_time += (end - start)

    avg_time = total_time / test_times * 1000
    fps = 1000 / avg_time
    return avg_time, fps


if __name__ == '__main__':
    model = load_model()

    print("\n" + "=" * 50)
    count_params_and_flops(model)
    print("=" * 50)

    avg_time, fps = test_inference_speed(model)

    print("\n" + "=" * 50)
    print(f"Avg Inference Time: {avg_time:.2f} ms")
    print(f" FPS：{fps:.2f}")
    print("=" * 50)