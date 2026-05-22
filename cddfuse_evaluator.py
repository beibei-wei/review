# 导入数值计算库
import os
import numpy as np
# 导入OpenCV图像处理库
import cv2
# 导入sklearn机器学习指标库
import sklearn.metrics as skm
# 导入二维卷积函数，用于梯度、滤波计算
from scipy.signal import convolve2d
# 导入数学库
import math
# 导入结构相似性指标函数
from skimage.metrics import structural_similarity as ssim
import torch.nn.functional as F
__all__ = ['Evaluator', 'image_read_cv2']

# 使用OpenCV读取图像，支持RGB、灰度、YCrCb三种格式
def image_read_cv2(path, mode='RGB'):
    # 以BGR格式读取并转为浮点型
    img_BGR = cv2.imread(path).astype('float32')
    assert mode == 'RGB' or mode == 'GRAY' or mode == 'YCrCb', 'mode error'
    if mode == 'RGB':
        # BGR转RGB
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        # 转为灰度图并四舍五入取整
        img = np.round(cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY))
    elif mode == 'YCrCb':
        # 转为亮度-色度空间
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
    return img

# 图像融合评价指标类，实现多种客观评价指标
'''
# --------------------------------------------
# 图像质量评估指标：EN / SD / SF / AG / MI / MSE / CC / PSNR / SCD / VIF / Qabf / SSIM /
# --------------------------------------------
'''
class Evaluator():
    # 输入校验：检查图像类型、形状、维度是否合法
    @classmethod
    def input_check(cls, imgF, imgA=None, imgB=None):
        if imgA is None:
            # 单图输入：检查是否为numpy数组、是否为二维灰度图
            assert type(imgF) == np.ndarray, 'type error'
            assert len(imgF.shape) == 2, 'dimension error'
        else:
            # 三图输入：检查类型一致、形状一致、均为灰度图
            assert type(imgF) == type(imgA) == type(imgB) == np.ndarray, 'type error'
            assert imgF.shape == imgA.shape == imgB.shape, 'shape error'
            assert len(imgF.shape) == 2, 'dimension error'

    # 信息熵 Entropy，反映图像信息丰富程度
    @classmethod
    def EN(cls, img):
        cls.input_check(img)
        a = np.uint8(np.round(img)).flatten()    #图像数据处理：四舍五入 → 转8位无符号整数 → 展平成一维数组
        h = np.bincount(a) / a.shape[0]          #统计每个像素值出现的频次，并计算概率分布
        return -sum(h * np.log2(h + (h == 0)))   #计算信息熵（核心公式）

    # 标准差 Standard Deviation，反映图像对比度
    @classmethod
    def SD(cls, img):
        cls.input_check(img)
        return np.std(img)                      # 调用 numpy 库的 std() 函数，计算图像数据的标准差

    # 空间频率 Spatial Frequency，反映图像清晰程度
    @classmethod
    def SF(cls, img):
        cls.input_check(img)
        return np.sqrt(np.mean((img[:, 1:] - img[:, :-1]) ** 2) + np.mean((img[1:, :] - img[:-1, :]) ** 2))

    # 平均梯度 Average Gradient，反映图像细节变化程度
    @classmethod
    def AG(cls, img):
        cls.input_check(img)
        Gx, Gy = np.zeros_like(img), np.zeros_like(img)

        #计算水平梯度
        Gx[:, 0] = img[:, 1] - img[:, 0]             # 左边界（第0列）处理
        Gx[:, -1] = img[:, -1] - img[:, -2]          # 右边界（第-1列）处理
        Gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2 # 内部计算
        #计算垂直梯度
        Gy[0, :] = img[1, :] - img[0, :]
        Gy[-1, :] = img[-1, :] - img[-2, :]
        Gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2
        return np.mean(np.sqrt((Gx ** 2 + Gy ** 2) / 2))

    # 互信息 Mutual Information，衡量融合图保留源图像信息多少
    @classmethod
    def MI(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        #  flatten 展平
        FA=skm.mutual_info_score(image_F.flatten(),image_A.flatten())
        FB=skm.mutual_info_score(image_F.flatten(),image_B.flatten())
        return FA+FB

    # 均方误差 MSE，内部辅助计算函数
    @classmethod
    def MSE(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return (np.mean((image_A - image_F) ** 2) + np.mean((image_B - image_F) ** 2)) / 2

    # 相关系数CC Correlation Coefficient，衡量融合图与源图线性相关程度
    @classmethod
    def CC(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        rAF = np.sum((image_A - np.mean(image_A)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        rBF = np.sum((image_B - np.mean(image_B)) * (image_F - np.mean(image_F))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((image_F - np.mean(image_F)) ** 2)))
        return (rAF + rBF) / 2

    # 峰值信噪比 PSNR
    @classmethod
    def PSNR(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return 10 * np.log10(np.max(image_F) ** 2 / cls.MSE(image_F, image_A, image_B))

    # 差异相关和SCD   Sum of Correlations of Differences
    @classmethod
    def SCD(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        imgF_A = image_F - image_A
        imgF_B = image_F - image_B
        corr1 = np.sum((image_A - np.mean(image_A)) * (imgF_B - np.mean(imgF_B))) / np.sqrt(
            (np.sum((image_A - np.mean(image_A)) ** 2)) * (np.sum((imgF_B - np.mean(imgF_B)) ** 2)))
        corr2 = np.sum((image_B - np.mean(image_B)) * (imgF_A - np.mean(imgF_A))) / np.sqrt(
            (np.sum((image_B - np.mean(image_B)) ** 2)) * (np.sum((imgF_A - np.mean(imgF_A)) ** 2)))
        return corr1 + corr2

    # 视觉信息保真度 VIFF
    @classmethod
    def VIFF(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        return cls.compare_viff(image_A, image_F)+cls.compare_viff(image_B, image_F)

    # 计算一对参考图与失真图的VIFF值
    @classmethod
    def compare_viff(cls,ref, dist):
        sigma_nsq = 2
        eps = 1e-10

        num = 0.0
        den = 0.0
        for scale in range(1, 5):

            N = 2 ** (4 - scale + 1) + 1
            sd = N / 5.0

            # 构造高斯核
            m, n = [(ss - 1.) / 2. for ss in (N, N)]
            y, x = np.ogrid[-m:m + 1, -n:n + 1]
            h = np.exp(-(x * x + y * y) / (2. * sd * sd))
            h[h < np.finfo(h.dtype).eps * h.max()] = 0
            sumh = h.sum()
            if sumh != 0:
                win = h / sumh

            if scale > 1:
                ref = convolve2d(ref, np.rot90(win, 2), mode='valid')
                dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
                ref = ref[::2, ::2]
                dist = dist[::2, ::2]

            mu1 = convolve2d(ref, np.rot90(win, 2), mode='valid')
            mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
            mu1_sq = mu1 * mu1
            mu2_sq = mu2 * mu2
            mu1_mu2 = mu1 * mu2
            sigma1_sq = convolve2d(ref * ref, np.rot90(win, 2), mode='valid') - mu1_sq
            sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
            sigma12 = convolve2d(ref * dist, np.rot90(win, 2), mode='valid') - mu1_mu2

            sigma1_sq[sigma1_sq < 0] = 0
            sigma2_sq[sigma2_sq < 0] = 0

            g = sigma12 / (sigma1_sq + eps)
            sv_sq = sigma2_sq - g * sigma12

            g[sigma1_sq < eps] = 0
            sv_sq[sigma1_sq < eps] = sigma2_sq[sigma1_sq < eps]
            sigma1_sq[sigma1_sq < eps] = 0

            g[sigma2_sq < eps] = 0
            sv_sq[sigma2_sq < eps] = 0

            sv_sq[g < 0] = sigma2_sq[g < 0]
            g[g < 0] = 0
            sv_sq[sv_sq <= eps] = eps

            num += np.sum(np.log10(1 + g * g * sigma1_sq / (sv_sq + sigma_nsq)))
            den += np.sum(np.log10(1 + sigma1_sq / sigma_nsq))

        vifp = num / den

        if np.isnan(vifp):
            return 1.0
        else:
            return vifp

    # Qabf 综合融合性能指标
    @classmethod
    def Qabf(cls, image_F, image_A, image_B):
        cls.input_check(image_F, image_A, image_B)
        gA, aA = cls.Qabf_getArray(image_A)
        gB, aB = cls.Qabf_getArray(image_B)
        gF, aF = cls.Qabf_getArray(image_F)
        QAF = cls.Qabf_getQabf(aA, gA, aF, gF)
        QBF = cls.Qabf_getQabf(aB, gB, aF, gF)

        # 计算最终Qabf
        deno = np.sum(gA + gB)
        nume = np.sum(np.multiply(QAF, gA) + np.multiply(QBF, gB))
        return nume / deno

    # Sobel算子计算梯度幅值和方向
    @classmethod
    def Qabf_getArray(cls,img):
        h1 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]]).astype(np.float32)
        h2 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]]).astype(np.float32)
        h3 = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).astype(np.float32)

        SAx = convolve2d(img, h3, mode='same')
        SAy = convolve2d(img, h1, mode='same')
        gA = np.sqrt(np.multiply(SAx, SAx) + np.multiply(SAy, SAy))
        aA = np.zeros_like(img)
        aA[SAx == 0] = math.pi / 2
        aA[SAx != 0]=np.arctan(SAy[SAx != 0] / SAx[SAx != 0])
        return gA, aA

    # 计算Qabf子项
    @classmethod
    def Qabf_getQabf(cls,aA, gA, aF, gF):
        L = 1
        Tg = 0.9994
        kg = -15
        Dg = 0.5
        Ta = 0.9879
        ka = -22
        Da = 0.8
        GAF,AAF,QgAF,QaAF,QAF = np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA),np.zeros_like(aA)
        GAF[gA>gF]=gF[gA>gF]/gA[gA>gF]
        GAF[gA == gF] = gF[gA == gF]
        GAF[gA <gF] = gA[gA<gF]/gF[gA<gF]
        AAF = 1 - np.abs(aA - aF) / (math.pi / 2)
        QgAF = Tg / (1 + np.exp(kg * (GAF - Dg)))
        QaAF = Ta / (1 + np.exp(ka * (AAF - Da)))
        QAF = QgAF* QaAF
        return QAF

    # 结构相似性 SSIM，分别计算融合图与可见光、红外图的SSIM并相加
    @classmethod
    def SSIM(cls, image_F, image_A, image_B,data_range=255):
        cls.input_check(image_F, image_A, image_B)
        return ssim(image_F,image_A, data_range=data_range)+ssim(image_F,image_B, data_range=data_range)

# 独立实现的VIFF计算函数，功能与类内VIFF一致
def VIFF(image_F, image_A, image_B):
    refA=image_A
    refB=image_B
    dist=image_F

    sigma_nsq = 2
    eps = 1e-10
    numA = 0.0
    denA = 0.0
    numB = 0.0
    denB = 0.0
    for scale in range(1, 5):
        N = 2 ** (4 - scale + 1) + 1
        sd = N / 5.0
        m, n = [(ss - 1.) / 2. for ss in (N, N)]
        y, x = np.ogrid[-m:m + 1, -n:n + 1]
        h = np.exp(-(x * x + y * y) / (2. * sd * sd))
        h[h < np.finfo(h.dtype).eps * h.max()] = 0
        sumh = h.sum()
        if sumh != 0:
            win = h / sumh

        if scale > 1:
            refA = convolve2d(refA, np.rot90(win, 2), mode='valid')
            refB = convolve2d(refB, np.rot90(win, 2), mode='valid')
            dist = convolve2d(dist, np.rot90(win, 2), mode='valid')
            refA = refA[::2, ::2]
            refB = refB[::2, ::2]
            dist = dist[::2, ::2]

        mu1A = convolve2d(refA, np.rot90(win, 2), mode='valid')
        mu1B = convolve2d(refB, np.rot90(win, 2), mode='valid')
        mu2 = convolve2d(dist, np.rot90(win, 2), mode='valid')
        mu1_sq_A = mu1A * mu1A
        mu1_sq_B = mu1B * mu1B
        mu2_sq = mu2 * mu2
        mu1A_mu2 = mu1A * mu2
        mu1B_mu2 = mu1B * mu2
        sigma1A_sq = convolve2d(refA * refA, np.rot90(win, 2), mode='valid') - mu1_sq_A
        sigma1B_sq = convolve2d(refB * refB, np.rot90(win, 2), mode='valid') - mu1_sq_B
        sigma2_sq = convolve2d(dist * dist, np.rot90(win, 2), mode='valid') - mu2_sq
        sigma12_A = convolve2d(refA * dist, np.rot90(win, 2), mode='valid') - mu1A_mu2
        sigma12_B = convolve2d(refB * dist, np.rot90(win, 2), mode='valid') - mu1B_mu2

        sigma1A_sq[sigma1A_sq < 0] = 0
        sigma1B_sq[sigma1B_sq < 0] = 0
        sigma2_sq[sigma2_sq < 0] = 0

        gA = sigma12_A / (sigma1A_sq + eps)
        gB = sigma12_B / (sigma1B_sq + eps)
        sv_sq_A = sigma2_sq - gA * sigma12_A
        sv_sq_B = sigma2_sq - gB * sigma12_B

        gA[sigma1A_sq < eps] = 0
        gB[sigma1B_sq < eps] = 0
        sv_sq_A[sigma1A_sq < eps] = sigma2_sq[sigma1A_sq < eps]
        sv_sq_B[sigma1B_sq < eps] = sigma2_sq[sigma1B_sq < eps]
        sigma1A_sq[sigma1A_sq < eps] = 0
        sigma1B_sq[sigma1B_sq < eps] = 0

        gA[sigma2_sq < eps] = 0
        gB[sigma2_sq < eps] = 0
        sv_sq_A[sigma2_sq < eps] = 0
        sv_sq_B[sigma2_sq < eps] = 0

        sv_sq_A[gA < 0] = sigma2_sq[gA < 0]
        sv_sq_B[gB < 0] = sigma2_sq[gB < 0]
        gA[gA < 0] = 0
        gB[gB < 0] = 0
        sv_sq_A[sv_sq_A <= eps] = eps
        sv_sq_B[sv_sq_B <= eps] = eps

        numA += np.sum(np.log10(1 + gA * gA * sigma1A_sq / (sv_sq_A + sigma_nsq)))
        numB += np.sum(np.log10(1 + gB * gB * sigma1B_sq / (sv_sq_B + sigma_nsq)))
        denA += np.sum(np.log10(1 + sigma1A_sq / sigma_nsq))
        denB += np.sum(np.log10(1 + sigma1B_sq / sigma_nsq))

    vifpA = numA / denA
    vifpB =numB / denB

    if np.isnan(vifpA):
        vifpA=1
    if np.isnan(vifpB):
        vifpB = 1
    return vifpA+vifpB


class Resize_16(object):
    def __init__(self):
        pass
    def __call__(self, image, target):
        # image, target 都是 cv2 打开的 numpy 数组 (H, W, C)
        height, width = image.shape[:2]  # cv2 图像取高宽用 shape

        new_width = (width // 16) * 16
        new_height = (height // 16) * 16
        new = min(new_width, new_height)
        # cv2 resize 用法
        image = cv2.resize(image, (new, new))
        target = cv2.resize(target, (new, new))

        return image, target

def find_image_path(folder, name_stem):
    exts = ['.png', '.bmp', '.jpg', '.jpeg', '.tiff']
    for ext in exts:
        path = os.path.join(folder, name_stem + ext)
        if os.path.exists(path):
            return path
    return None

