import csv
import os

def csv_to_txt(csv_file, txt_file, n=2):
    with open(csv_file, 'r', encoding='utf-8') as infile, \
            open(txt_file, 'w', encoding='utf-8') as outfile:
        reader = csv.DictReader(infile)
        for row in reader:
            # 处理评论中的换行符并拼接格式
            comment = row['评论'].replace('\n', ' ')
            outfile.write(f"{n} {comment}\n")


if __name__ == '__main__':
    # 设置输入文件路径
    csv_file = r"D:\PyCharm 2024.1.7\MyProj\JDcomment\commentDate\comments_100142621616_差评.csv"

    # 创建输出目录（若不存在）
    output_dir = "txtDate"
    os.makedirs(output_dir, exist_ok=True)  # [9,10](@ref)

    # 生成输出路径（保持原文件名，仅改扩展名）
    base_name = os.path.splitext(os.path.basename(csv_file))[0]
    txt_file = os.path.join(output_dir, f"{base_name}.txt")  # [9,10](@ref)

    # 执行转换
    csv_to_txt(csv_file, txt_file)
    print(f"转换完成！输出文件：{txt_file}")

