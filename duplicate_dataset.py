import os
import shutil
from pathlib import Path
import sys
import random

def duplicate_dataset(source_dir, folder_copies=100, image_copies=5):
    """
    复制数据集的类别文件夹和文件夹内的图片
    
    参数:
    source_dir: 源目录路径
    folder_copies: 每个类别文件夹要复制的次数
    image_copies: 每个图片要复制的次数
    """
    # 确保源目录存在
    source_path = Path(source_dir)
    if not source_path.exists() or not source_path.is_dir():
        print(f"错误：源目录 {source_dir} 不存在!")
        sys.exit(1)
    
    # 获取所有子文件夹（类别文件夹）
    class_dirs = [d for d in source_path.iterdir() if d.is_dir()]
    
    # 显示找到的文件夹
    print(f"在 {source_dir} 中找到 {len(class_dirs)} 个类别文件夹")
    
    # 第1步：复制并扩充每个原始类别文件夹内的图片
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"处理原始类别: {class_name}")
        
        # 获取该类别下的所有图片
        images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']]
        
        if not images:
            print(f"  警告: {class_name} 文件夹中没有找到图片文件")
            continue
            
        print(f"  在 {class_name} 中找到 {len(images)} 张图片")
        
        # 复制每张图片
        for i, img in enumerate(images):
            for j in range(1, image_copies + 1):
                # 创建新的文件名
                new_name = f"{img.stem}_copy_{j}{img.suffix}"
                new_path = class_dir / new_name
                
                # 复制图片
                shutil.copy2(img, new_path)
                
                # 每复制50张图片打印一次进度
                if (i * image_copies + j) % 50 == 0:
                    print(f"  已在 {class_name} 中复制 {i * image_copies + j} 张图片")
    
    # 第2步：复制类别文件夹(已包含原始和复制的图片)
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"复制扩充后的类别文件夹: {class_name}")
        
        # 复制该文件夹100次
        for i in range(1, folder_copies + 1):
            new_dir = Path(f"{class_dir}_copy_{i}")
            print(f"  创建文件夹复制 {i}: {new_dir}")
            shutil.copytree(class_dir, new_dir)
    
    print("所有复制操作完成!")

if __name__ == "__main__":
    # 源目录路径
    # source_directory = "minc-2500-tiny/train"
    # source_directory = "minc-2500-tiny/test"
    source_directory = "minc-2500-tiny/val"
    
    # 执行复制
    duplicate_dataset(source_directory, folder_copies=500, image_copies=40)
