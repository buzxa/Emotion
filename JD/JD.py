# 导入自动化模块
from DrissionPage import ChromiumPage
# 模拟键盘操作
from DrissionPage.common import Keys
# 导入时间模块
import time
# 导入csv模块和os模块
import csv
import os

# 创建保存目录（与源代码同级）1
save_dir = os.path.join(os.path.dirname(__file__), 'goodsDate')
os.makedirs(save_dir, exist_ok=True)

# 构造文件完整路径
file_path = os.path.join(save_dir, 'data.csv')

# 创建文件对象
f = open(file_path, mode='w', encoding='utf-8', newline='')
# 字典写入的方法
csv_writer = csv.DictWriter(f, fieldnames=[
    '标题',
    '价格',
    '评价数',
    '店铺',
    '详情页',
])
# 写入表头
csv_writer.writeheader()

# 实例化浏览器对象
dp = ChromiumPage()
# 访问网站
dp.get('https://www.jd.com/')
# 定位搜素框, 输入内容
dp.ele('css:#key').input('iphone16')
# 回车按钮
dp.ele('css:#key').input(Keys.ENTER)
# 延时等待
time.sleep(2)

# for循环翻页
for page in range(1, 11):
    dp.scroll.to_bottom()
    # 第一次提取, 提取所有商品所在li标签
    lis = dp.eles('css:.gl-item')
    # for循环遍历, 提取列表里面元素
    for li in lis:
        try:
            title = li.ele('css:.p-name a em').text  # 标题
            href = li.ele('css:.p-name a').attr('href')  # 详情页
            price = li.ele('css:.p-price i').text  # 售价
            commit = li.ele('css:.p-commit a').text  # 评价数
            shop_name = li.ele('css:.hd-shopname').text  # 店铺
            dit = {
                '标题': title,
                '价格': price,
                '评价数': commit,
                '店铺': shop_name,
                '详情页': href,
            }
            # 写入数据
            csv_writer.writerow(dit)
            print(dit)
        except:
            pass
    # 点击翻页
    lis[0].input(Keys.RIGHT)

# 关闭文件
f.close()
