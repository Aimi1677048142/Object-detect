import os

import os


def get_directory_structure(root_dir):
    dir_dict = {}

    # 遍历指定目录及其所有子目录
    for dirpath, _, filenames in os.walk(root_dir):
        # 获取相对于根目录的相对路径
        relative_path = os.path.relpath(dirpath, root_dir)

        # 如果当前路径是根目录 itself, 跳过它
        if relative_path == ".":
            continue

        # 将相对路径作为字典的键，文件列表作为值
        dir_dict[relative_path] = filenames

    return dir_dict


# 使用示例
root_dir = r"D:\pcb_data\train_data"
directory_structure = get_directory_structure(root_dir)

# 输出字典结构
for directory, files in directory_structure.items():
    print(f"Directory: {directory}")
    join = os.path.join(root_dir,directory, files[0])
    print(join)
    print(f"Files: {files}")
    # print()
    break

    # print()
