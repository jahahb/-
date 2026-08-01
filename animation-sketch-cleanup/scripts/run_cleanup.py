import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from PIL import Image

# ======================= 路径配置 =======================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
INPUT_DIR = os.path.join(BASE_DIR, "data", "test_input")
GT_DIR = os.path.join(BASE_DIR, "data", "gt_clean")
OUTPUT_DIR = os.path.join(BASE_DIR, "data", "model_output")
COMPARE_DIR = os.path.join(BASE_DIR, "result_compare")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(COMPARE_DIR, exist_ok=True)


def read_image_support_tga(img_path):
    """兼容TGA/JPG/PNG图片读取，统一输出OpenCV BGR数组"""
    pil_img = Image.open(img_path).convert("RGB")
    rgb_np = np.array(pil_img)
    return cv2.cvtColor(rgb_np, cv2.COLOR_RGB2BGR)


def sketch_clean(image_bgr):
    """动画草图自动描原核心函数"""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    # 筛选黑色有效轮廓线
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 245, 75])
    black_mask = cv2.inRange(hsv, lower_black, upper_black)

    # 形态学降噪
    kernel_small = np.ones((1, 1), np.uint8)
    black_mask = cv2.morphologyEx(black_mask, cv2.MORPH_CLOSE, kernel_small)

    # 骨架提取，线条细化
    mask_norm = black_mask.astype(np.float32) / 255.0
    skeleton = skeletonize(mask_norm)
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)
    return skeleton_uint8


def concat_compare(sketch_img, pred_img, gt_img, save_path):
    """拼接三连对比图：草图｜算法输出｜真值清线稿"""
    pred_3ch = cv2.cvtColor(pred_img, cv2.COLOR_GRAY2BGR)
    combine = np.hstack([sketch_img, pred_3ch, gt_img])
    cv2.imwrite(save_path, combine)


if __name__ == "__main__":
    # 构建GT映射字典：{文件名主干 : 文件全名}
    gt_map = {}
    for fname in os.listdir(GT_DIR):
        stem, ext = os.path.splitext(fname)
        gt_map[stem] = fname

    input_list = [f for f in os.listdir(INPUT_DIR) if f.endswith((".png", ".jpg", ".jpeg"))]
    print(f"检测到测试样本数量：{len(input_list)}")

    for filename in input_list:
        stem, ext = os.path.splitext(filename)
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        compare_path = os.path.join(COMPARE_DIR, filename)

        # 根据主干名称匹配，忽略后缀
        if stem not in gt_map:
            print(f"警告：{filename} 缺失对应gt真值图，跳过")
            continue

        gt_path = os.path.join(GT_DIR, gt_map[stem])
        # 读取图片（兼容TGA）
        img_sketch = cv2.imread(input_path)
        img_gt = read_image_support_tga(gt_path)

        clean_line = sketch_clean(img_sketch)
        cv2.imwrite(output_path, clean_line)
        concat_compare(img_sketch, clean_line, img_gt, compare_path)
        print(f"✅ 处理完成：{filename}")

    print("\n=====全部推理结束=====")
    print(f"算法输出目录：{OUTPUT_DIR}")
    print(f"三连对比图目录：{COMPARE_DIR}")