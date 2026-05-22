import os
import math
from PIL import Image, ImageDraw, ImageFont
import matplotlib
matplotlib.use('Agg')



root_dir = "./data"                       # The root directory of all method folders
target_filename = "00004N"                # Target image
scale = 8                                 # Magnification factor
zoom_corner = "right_bottom"              # right_bottom/right_top/left_bottom/left_top
save_dir ="./"
date=['MSRS']

roi_list =["00004N", 103, 194, 200, 283, 3,  82, 194, 160, 283,"right_bottom"]

selected = roi_list
target_filename = selected[0]  # 文件名是第1个元素
x1, y1, x2, y2 = selected[1], selected[2], selected[3], selected[4]
scale = selected[5]
x1_2, y1_2, x2_2, y2_2 = selected[6], selected[7], selected[8], selected[9]
zoom_corner = selected[10]

gap = 10
border = 20
title_height = 60


DPI = 300
canvas_w_mm = 192
canvas_h_mm = 53.5
canvas_w = round(canvas_w_mm * DPI / 25.4)
canvas_h = round(canvas_h_mm * DPI / 25.4)


os.makedirs(save_dir, exist_ok=True)
valid_images = []
method_names = []

ordered_methods = [
    "MSRS_VI",
    "MSRS_IR",
    "IASSF",
    "EMMA",
    "T2EA",
    "DCEvo",
    "CDDFuse",
    "ITFuse",
    "S4Fusion",
    "CAWM-Mamba",
    "Wavelet-Mamba",
    "MKDFusion",
    "SFMFusion",
    "F2Fusion"
]


for idx, method in enumerate(ordered_methods):
    method_path = os.path.join(root_dir, method )
    if not os.path.isdir(method_path):
        continue

    img_path = None
    for ext in ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']:
        temp_path = os.path.join(root_dir, method , target_filename + ext)
        if os.path.exists(temp_path):
            img_path = temp_path
            break
    if not img_path:
        print(f"Skip {method}: no {target_filename}")
        continue

    img = Image.open(img_path).convert("RGB")
    W, H = img.size
    draw = ImageDraw.Draw(img)

    if method == "CAWM-Mamba":
        rx1, ry1, rx2, ry2 = x1_2, y1_2, x2_2, y2_2
    else:
        rx1, ry1, rx2, ry2 = x1, y1, x2, y2

    crop = img.crop((rx1, ry1, rx2, ry2))
    zoom = crop.resize(((rx2-rx1)*scale, (ry2-ry1)*scale), Image.BICUBIC)
    zw, zh = zoom.size

    draw_zoom = ImageDraw.Draw(zoom)
    draw_zoom.rectangle([0, 0, zw-1, zh-1], outline="yellow", width=3)

    if zoom_corner == "right_bottom":
        px, py = W - zw, H - zh
    elif zoom_corner == "right_top":
        px, py = W - zw, 0
    elif zoom_corner == "left_bottom":
        px, py = 0, H - zh
    else:
        px, py = 0, 0

    img.paste(zoom, (px, py))
    draw.rectangle([rx1, ry1, rx2, ry2], outline="yellow", width=3)

    valid_images.append(img)
    method_names.append(method)

if not valid_images:
    print("No valid images!")
    exit()

N = len(valid_images)
cols = min(6, N)
rows = math.ceil(N / cols)

canvas = Image.new("RGB", (canvas_w, canvas_h), "white")
draw = ImageDraw.Draw(canvas)
font = ImageFont.load_default(size=24)


total_w = canvas_w - 2 * border - (cols - 1) * gap
total_h = canvas_h - 2 * border - rows * title_height - (rows - 1) * gap
cell_w = total_w / cols
cell_h = total_h / rows


for idx, (img, method) in enumerate(zip(valid_images, method_names)):
    row = idx // cols
    col = idx % cols

    ow, oh = img.size

    scale_ratio = cell_h / oh
    nw = int(ow * scale_ratio)
    nh = int(cell_h)

    img = img.resize((nw, nh), Image.BICUBIC)

    x = int(border + col * (cell_w + gap) + (cell_w - nw) / 2)
    y = int(border + row * (cell_h + title_height + gap) + title_height)
    canvas.paste(img, (x, y))



save_path = os.path.join(save_dir, f"{target_filename}.png")
canvas.save(save_path, dpi=(300, 300))


print("All done! Saved to:", save_path)