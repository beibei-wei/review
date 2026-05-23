import os
import numpy as np
from cddfuse_evaluator import image_read_cv2, Resize_16, find_image_path, Evaluator

if __name__ == '__main__':
 for dataset_name in ["MSRS"]:      # dataset name
     print("\n" * 2 + "=" * 80)
     for model_name in ["CAWM-Mamba", "CDDFuse", "DCEvo", "EMMA", "F2Fusion", "IASSF", "ITFuse", "MKDFusion", "origin_ir",
                        "origin_vi", "S4Fusion", "SFMFusion", "T2EA", "Wavelet-Mamba"]:       # method name
         print("The test result of " + dataset_name + ' :')

         origin_ir = "./data/origin_ir"                 # Original test image saving address
         origin_vi = "./data/origin_vi"
         fus_dir = os.path.join("./data",model_name)    # fused image storage address
         metric_result = np.zeros((12))
         print(fus_dir)

         for img_name in os.listdir(origin_ir):
             ir = image_read_cv2(os.path.join(origin_ir, img_name), 'GRAY')
             vi = image_read_cv2(os.path.join(origin_vi, img_name), 'GRAY')

             Resize_161=Resize_16()
             ir,vi = Resize_161(ir,vi)
             ir = np.clip(ir, 0, 255).astype(np.uint8).astype(np.float32)
             vi = np.clip(vi, 0, 255).astype(np.uint8).astype(np.float32)
             img_stem = img_name.split('.')[0]
             img_path = find_image_path(fus_dir, img_stem)
             fi = image_read_cv2(img_path, 'GRAY')

            # EN / SD / SF / AG / MI / MSE / CC / PSNR / SCD / VIF / Qabf / SSIM /

             metric_result += np.array([
                 Evaluator.EN(fi),
                 Evaluator.SD(fi),
                 Evaluator.SF(fi),
                 Evaluator.AG(fi),
                 Evaluator.MI(fi, ir, vi),
                 Evaluator.MSE(fi, ir, vi),
                 Evaluator.CC(fi, ir, vi),
                 Evaluator.PSNR(fi, ir, vi),
                 Evaluator.SCD(fi, ir, vi),
                 Evaluator.VIFF(fi, ir, vi),
                 Evaluator.Qabf(fi, ir, vi),
                 Evaluator.SSIM(fi, ir, vi, data_range=255)])


         metric_result /= len(os.listdir(fus_dir))


         print(f"{'':<10} {'EN':<10} {'SD':<10} {'SF':<10} {'AG':<10} {'MI':<10} {'MSE':<12} "
               f"{'CC':<10} {'PSNR':<10} {'SCD':<10} {'VIF':<10} {'Qabf':<10} {'SSIM':<10}")

         print(f"{model_name:<10} {np.round(metric_result[0], 4):<10} "
               f"{np.round(metric_result[1], 4):<10} {np.round(metric_result[2], 4):<10} "
               f"{np.round(metric_result[3], 4):<10} {np.round(metric_result[4], 4):<10} "
               f"{np.round(metric_result[5], 4):<12} {np.round(metric_result[6], 4):<10} "
               f"{np.round(metric_result[7], 4):<10} {np.round(metric_result[8], 4):<10} "
               f"{np.round(metric_result[9], 4):<10} {np.round(metric_result[10], 4):<10} "
               f"{np.round(metric_result[11], 4):<10}")
         print("=" * 80)