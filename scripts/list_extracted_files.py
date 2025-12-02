#!/usr/bin/env python3
"""
遍历 /mnt/data/user/shao_bing/src_extracted/ 下所有文件，输出绝对路径到 list.txt。
"""
import os

DATA_ROOT = "/mnt/data/user/shao_bing/src_extracted/"
OUTPUT_FILE = "list.txt"

def main():
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for dirpath, _, filenames in os.walk(DATA_ROOT):
            for name in filenames:
                abs_path = os.path.join(dirpath, name)
                f.write(abs_path + "\n")
    print(f"已保存所有文件路径到 {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
