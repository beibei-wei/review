import torch
import time
import torch.nn.functional as F
from models.IASSF import RestormerUNet, iassf


device = "cuda" if torch.cuda.is_available() else "cpu"
input_h = 480
input_w = 640
warmup = 10
test_times = 100



def pad_to_multiple(x, multiple=32):
    B, C, H, W = x.shape
    new_h = ((H + multiple - 1) // multiple) * multiple
    new_w = ((W + multiple - 1) // multiple) * multiple
    pad_h = new_h - H
    pad_w = new_w - W
    x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h), mode='reflect')
    return x, H, W


def load_models():
    dehaze_net = iassf().to(device)
    fusion_net = RestormerUNet().to(device)

    saved_model_dir = 'saved_models/iassf/IASSF.pth'                 ## set weights path
    state_dict = torch.load(saved_model_dir, map_location=device)

    def load_model_state(model, state_dict):
        model_state_dict = model.state_dict()
        new_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state_dict and v.size() == model_state_dict[k].size():
                new_state_dict[k] = v
        model_state_dict.update(new_state_dict)
        model.load_state_dict(model_state_dict)

    load_model_state(dehaze_net, state_dict['dehaze_network_state_dict'])
    load_model_state(fusion_net, state_dict['fusion_network_state_dict'])

    dehaze_net.eval()
    fusion_net.eval()
    return dehaze_net, fusion_net



def count_total_params(dehaze_net, fusion_net):
    params_dehaze = sum(p.numel() for p in dehaze_net.parameters())
    params_fusion = sum(p.numel() for p in fusion_net.parameters())
    total = params_dehaze + params_fusion

    print(f" Params: {total / 1e6:.2f} M")
    return total


def test_speed(dehaze_net, fusion_net):
    print(f"warmup  {warmup} ...")
    with torch.no_grad():
        for _ in range(warmup):
            vi = torch.randn(1, 1, input_h, input_w).to(device)
            ir = torch.randn(1, 1, input_h, input_w).to(device)

            vi, _, _ = pad_to_multiple(vi)
            ir, _, _ = pad_to_multiple(ir)


            vi = vi.repeat(1, 3, 1, 1)
            ir = ir.repeat(1, 3, 1, 1)

            feat, ir_feat = dehaze_net(vi, ir)
            fusion_net(feat, ir_feat)

    print(f"test times: {test_times}  ...")
    total_time = 0.0
    with torch.no_grad():
        for _ in range(test_times):
            vi = torch.randn(1, 1, input_h, input_w).to(device)
            ir = torch.randn(1, 1, input_h, input_w).to(device)

            vi, _, _ = pad_to_multiple(vi)
            ir, _, _ = pad_to_multiple(ir)


            vi = vi.repeat(1, 3, 1, 1)
            ir = ir.repeat(1, 3, 1, 1)

            torch.cuda.synchronize()
            t0 = time.time()

            feat, ir_feat = dehaze_net(vi, ir)
            fusion_net(feat, ir_feat)

            torch.cuda.synchronize()
            t1 = time.time()
            total_time += t1 - t0

    avg_ms = total_time / test_times * 1000
    fps = 1000 / avg_ms
    return avg_ms, fps


if __name__ == '__main__':
    dehaze_net, fusion_net = load_models()

    print("\n" + "=" * 60)
    count_total_params(dehaze_net, fusion_net)
    print("=" * 60)

    avg_time, fps = test_speed(dehaze_net, fusion_net)

    print("\n" + "=" * 60)
    print(f"Avg Inference Time: {avg_time:.2f} ms")
    print(f" FPS：{fps:.2f}")
    print("=" * 60)