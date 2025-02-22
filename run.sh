#!/bin/bash

dir=/home/zwk/dataset/train

# 确保目录路径以斜杠结尾
if [[ "$dir" != */ ]]; then
    dir="$dir/"
fi

# 遍历所有 .tar 文件
for x in "$dir"*.tar; do
    if [ -f "$x" ]; then
        # 获取文件名（去掉后缀）
        filename=$(basename "$x" .tar)
        # 创建目标目录
        mkdir -p "$dir$filename"
        # 解压文件到目标目录
        tar -xvf "$x" -C "$dir$filename"
        echo "解压完成: $x -> $dir$filename"
        # 删除解压后的 .tar 文件
        rm "$x"
        echo "已删除: $x"
    fi
done

echo "所有 .tar 文件已解压并删除。"
