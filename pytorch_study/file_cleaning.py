import os
from PIL import Image


def delete_small_images(directory, min_size=250):
    # 遍历目录中的所有文件
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)

        # 检查文件是否是图片
        if os.path.isfile(file_path) and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp')):
            try:
                # 使用上下文管理器打开图片
                with Image.open(file_path) as img:
                    width, height = img.size
                    # 判断图片的宽和高是否都小于指定大小
                    if width < min_size and height < min_size:
                        print(f"Deleting {filename} (size: {width}x{height})")
                        img.close()  # 确保文件关闭
                        os.remove(file_path)
            except Exception as e:
                print(f"Error processing {filename}: {e}")


# 示例使用
directory = r'D:\pcb_data\yinziqi\yinziqi_image _copy'  # 替换为你的目录路径
delete_small_images(directory)