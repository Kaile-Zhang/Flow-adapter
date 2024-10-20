import numpy as np
import torch
import matplotlib.pyplot as plt
import os
from PIL import Image, ImageDraw, ImageFont
import cv2


UNKNOWN_FLOW_THRESH = 1e7

def make_color_wheel():
    """
    Generate color wheel according Middlebury color code
    :return: Color wheel
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR

    colorwheel = np.zeros([ncols, 3])

    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.transpose(np.floor(255*np.arange(0, RY) / RY))
    col += RY

    # YG
    colorwheel[col:col+YG, 0] = 255 - np.transpose(np.floor(255*np.arange(0, YG) / YG))
    colorwheel[col:col+YG, 1] = 255
    col += YG

    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.transpose(np.floor(255*np.arange(0, GC) / GC))
    col += GC

    # CB
    colorwheel[col:col+CB, 1] = 255 - np.transpose(np.floor(255*np.arange(0, CB) / CB))
    colorwheel[col:col+CB, 2] = 255
    col += CB

    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.transpose(np.floor(255*np.arange(0, BM) / BM))
    col += + BM

    # MR
    colorwheel[col:col+MR, 2] = 255 - np.transpose(np.floor(255 * np.arange(0, MR) / MR))
    colorwheel[col:col+MR, 0] = 255

    return colorwheel

def compute_color(u, v):
    """
    compute optical flow color map
    :param u: optical flow horizontal map
    :param v: optical flow vertical map
    :return: optical flow in color code
    """
    [h, w] = u.shape
    img = np.zeros([h, w, 3])
    nanIdx = np.isnan(u) | np.isnan(v)
    u[nanIdx] = 0
    v[nanIdx] = 0

    colorwheel = make_color_wheel()
    ncols = np.size(colorwheel, 0)

    rad = np.sqrt(u**2+v**2)

    a = np.arctan2(-v, -u) / np.pi

    fk = (a+1) / 2 * (ncols - 1) + 1

    k0 = np.floor(fk).astype(int)

    k1 = k0 + 1
    k1[k1 == ncols+1] = 1
    f = fk - k0

    for i in range(0, np.size(colorwheel,1)):
        tmp = colorwheel[:, i]
        col0 = tmp[k0-1] / 255
        col1 = tmp[k1-1] / 255
        col = (1-f) * col0 + f * col1

        idx = rad <= 1
        col[idx] = 1-rad[idx]*(1-col[idx])
        notidx = np.logical_not(idx)

        col[notidx] *= 0.75
        img[:, :, i] = np.uint8(np.floor(255 * col*(1-nanIdx)))

    return img

def flow_to_image(flow, display=False):
    """
    Convert flow into middlebury color code image
    :param flow: optical flow map
    :return: optical flow image in middlebury color
    """
    u = flow[:, :, 0]
    v = flow[:, :, 1]

    maxu = -999.
    maxv = -999.
    minu = 999.
    minv = 999.

    idxUnknow = (abs(u) > UNKNOWN_FLOW_THRESH) | (abs(v) > UNKNOWN_FLOW_THRESH)
    u[idxUnknow] = 0
    v[idxUnknow] = 0

    maxu = max(maxu, np.max(u))
    minu = min(minu, np.min(u))

    maxv = max(maxv, np.max(v))
    minv = min(minv, np.min(v))

    rad = np.sqrt(u ** 2 + v ** 2)
    maxrad = max(-1, np.max(rad))

    if display:
        print("max flow: %.4f\nflow range:\nu = %.3f .. %.3f\nv = %.3f .. %.3f" % (maxrad, minu,maxu, minv, maxv))

    u = u/(maxrad + np.finfo(float).eps)
    v = v/(maxrad + np.finfo(float).eps)

    img = compute_color(u, v)

    idx = np.repeat(idxUnknow[:, :, np.newaxis], 3, axis=2)
    img[idx] = 0

    return np.uint8(img)

def flow_to_file(flow, filepath):
    flow = flow.cpu().numpy()
    img = flow_to_image(flow)
    flow_flies = os.listdir(filepath)
    if flow_flies:
        flow_flies = [int(f.split('.')[0]) for f in flow_flies]
        idx = max(flow_flies) + 1
    else:
        idx = 1
    plt.imsave(os.path.join(filepath, f"{idx}.png"), img)

def show_standard_flow():
    flow_x = torch.arange(0, 256).unsqueeze(0).repeat(256, 1) - 128
    flow_y = torch.arange(0, 256).unsqueeze(1).repeat(1, 256) - 128
    flow = torch.stack((flow_x, flow_y), dim=-1).numpy()
    img = flow_to_image(flow)
    plt.imsave('flow.png', img)

def flow_show():
    flow_dir = 'flow'
    flow_files = os.listdir(flow_dir)
    flow_files.sort(key = lambda x: int(x.split('.')[0]))
    step_flow_num = 16
    begin_flow = 3200
    flow_step_files = [flow_files[i:i + step_flow_num] for i in range(begin_flow, len(flow_files), step_flow_num)]

    img_size = (64, 64)
    spacing = 10
    new_image_width = img_size[0] * 8 + spacing * (8 - 1)  # 8 列图片，7 个间距
    new_image_height = img_size[1] * 2 + spacing * (2 - 1)  # 2 行图片，1 个间距
    font = ImageFont.load_default(20)
    text_area_height = 50
    new_image_height = new_image_height + text_area_height
    # 在顶部添加文字
    text_position = (10, 10)  # 文字位置，左上角(10, 10)像素
    text_color = (0, 0, 0)  # 黑色文字

    # 调整每张图片到 64x64 并展示
    for j in range(len(flow_step_files)):
        new_image = Image.new('RGB', (new_image_width, new_image_height), 'white')
        # 创建一个 ImageDraw 对象以便在图像上绘制文字
        draw = ImageDraw.Draw(new_image)
        draw.text(text_position, text=f"step={j+1}", font=font, fill=text_color)
        image_paths = [os.path.join(flow_dir, ff) for ff in flow_step_files[j]]
        for i, image_path in enumerate(image_paths):
            img = Image.open(image_path)
            img_resized = img.resize(img_size)
            # 计算当前图片的位置
            row = i // 8  # 行索引
            col = i % 8   # 列索引
            x = col * (img_size[0] + spacing)  # x 坐标
            y = text_area_height + row * (img_size[1] + spacing)  # y 坐标
            # 将图片粘贴到新图像的相应位置
            new_image.paste(img_resized, (x, y))

        # 保存拼接后的图片
        new_image.save(f'flow-dog/{j+1}.png')

def flow_video(flow_dir):
    image_paths = os.listdir(flow_dir)
    image_paths.sort(key = lambda x: int(x.split('.')[0]))
    image_paths = [os.path.join(flow_dir, i) for i in image_paths]

    # 视频参数
    fps = 5  # 帧率（每秒10帧）
    video_duration = 10  # 视频时长5秒
    frame_size = (582, 188)  # 定义视频帧大小 (宽度, 高度)

    # 创建视频写入对象
    out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, frame_size)

    # 遍历每张图片并写入视频
    for image_path in image_paths:
        # 打开图片并调整到目标大小 (如果需要)
        img = Image.open(image_path)
        img_resized = img.resize(frame_size)
        
        # 转换为 OpenCV 格式
        img_cv = cv2.cvtColor(np.array(img_resized), cv2.COLOR_RGB2BGR)
        
        # 写入视频帧
        out.write(img_cv)

    # 释放视频写入对象
    out.release()
    print("视频已保存为 output_video.mp4")


if __name__ == "__main__":
    flow_show()
    flow_video('flow-dog')
