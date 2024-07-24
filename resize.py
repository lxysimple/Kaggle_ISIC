import os
from PIL import Image
import pandas as pd

def resize_and_save_images(file_paths, output_folder, size=(256, 256)):
    """
    将列表中的文件路径指向的图片缩放至指定尺寸并保存到指定文件夹下。

    参数:
        file_paths (list): 包含图片文件路径的列表。
        output_folder (str): 输出文件夹的路径。
        size (tuple): 缩放后的尺寸，默认为 (256, 256)。
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_path in file_paths:
        try:
            with Image.open(file_path) as img:
                img_resized = img.resize(size, Image.ANTIALIAS)
                # 获取原始文件名
                file_name = os.path.basename(file_path)
                # 构建输出文件路径
                output_path = os.path.join(output_folder, file_name)
                # 保存缩放后的图片
                img_resized.save(output_path)
                print(f"Saved resized image to {output_path}")
        except Exception as e:
            print(f"Failed to process {file_path}: {e}")


def get_paths():
    df = pd.read_csv(f"/home/xyli/kaggle/isicdir/others.csv")
    df_positive = df[df["benign_malignant"] == 'malignant'].reset_index()
    df_negative = df[df["benign_malignant"] == 'benign'].reset_index()
    # 保持一定的正负比例，不能让其失衡
    df_negative = self.df_negative[:len(self.df_positive)*20]
    isic_ids_positive = self.df_positive['isic_id'].values
    isic_ids_negative = self.df_negative['isic_id'].values

    paths = []
    for i in isic_ids_positive:
        path = f"/home/xyli/kaggle/isicdir/images/{isic_id}.jpg"
        paths.append(path)

    for i in isic_ids_negative:
        path = f"/home/xyli/kaggle/isicdir/images/{isic_id}.jpg"
        paths.append(path)
    
    return paths

# 示例文件路径列表
file_paths = get_paths()

# 指定输出文件夹
output_folder = '/home/xyli/kaggle/data_others/train-image'

# 调用函数进行处理
resize_and_save_images(file_paths, output_folder)
