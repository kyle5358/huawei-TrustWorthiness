#!/bin/bash

# 设置要处理的根目录
root_directory="/path/to/your/root/directory"

# 切换到根目录
cd "$root_directory"

# 遍历一级子目录
for first_level_dir in */; do
    # 检查是否是目录
    if [ -d "$first_level_dir" ]; then
        # 进入一级子目录
        cd "$first_level_dir"
        
        # 检查是否存在二级子目录
        if [ -d "images" ]; then
            # 移动所有图片到一级子目录
            mv images/* .
            
            # 删除空的二级子目录
            rmdir images
        fi
        
        # 返回到根目录
        cd "$root_directory"
    fi
done
