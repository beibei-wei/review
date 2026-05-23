import torch
import time
from thop import profile
from network.TEM import Taylor_Encoder
from network.FusionNet import FusionModel


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_h = 480
input_w = 640
warmup = 10
test_times = 20


def load_models():

    Net = Taylor_Encoder().to(device).eval()
    Fusion = FusionModel().to(device).eval()

    return Net, Fusion


def count_params(model, name):
    total = sum(p.numel() for p in model.parameters())
    print(f"  {name}：")
    print(f" Params：{total / 1e6:.2f} M")
    return total


def calculate_flops(Net, Fusion):
    with torch.no_grad():
        dummy_vis = torch.randn(1, 1, input_h, input_w).to(device)
        dummy_ir = torch.randn(1, 1, input_h, input_w).to(device)

        macs_enc, _ = profile(Net, inputs=(dummy_vis, 2), verbose=False)
        _, feat_vis = Net(dummy_vis, 2)
        _, feat_ir = Net(dummy_ir, 2)

        macs_fus, _ = profile(Fusion, inputs=(feat_vis, feat_ir), verbose=False)

        total_flops = 2 * (macs_enc + macs_fus) / 1e9
        print(f" {total_flops:.2f} GFLOPs")
    return total_flops


def test_speed(Net, Fusion):
    print(f"warmup {warmup}  ...")
    with torch.no_grad():
        for _ in range(warmup):
            dummy_vis = torch.randn(1, 1, input_h, input_w).to(device)
            dummy_ir = torch.randn(1, 1, input_h, input_w).to(device)
            _, feat_vis = Net(dummy_vis, 2)
            _, feat_ir = Net(dummy_ir, 2)
            Fusion(feat_vis, feat_ir)

    print(f"test times: {test_times} ...")
    total_time = 0
    with torch.no_grad():
        for _ in range(test_times):
            dummy_vis = torch.randn(1, 1, input_h, input_w).to(device)
            dummy_ir = torch.randn(1, 1, input_h, input_w).to(device)

            torch.cuda.synchronize()
            start = time.time()

            _, feat_vis = Net(dummy_vis, 2)
            _, feat_ir = Net(dummy_ir, 2)
            Fusion(feat_vis, feat_ir)

            torch.cuda.synchronize()
            end = time.time()
            total_time += end - start

    avg_time = total_time / test_times * 1000
    fps = 1000 / avg_time
    return avg_time, fps


if __name__ == '__main__':
    Net, Fusion = load_models()

    print("\n" + "="*50)
    params_net = count_params(Net, "Taylor")
    params_fus = count_params(Fusion, "Fusion")
    total_params = (params_net + params_fus) / 1e6
    print(f"Params: {total_params:.2f} M")
    calculate_flops(Net, Fusion)
    print("="*50)

    avg_time, fps = test_speed(Net, Fusion)
    print("\n" + "="*50)
    print(f"Avg Inference Time: {avg_time:.2f} ms/张")
    print(f"FPS：{fps:.2f}")
    print("="*50)