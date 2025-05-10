from DrissionPage import ChromiumPage
import time
import os  # 新增导入os模块
import csv
from tqdm import tqdm


def get_jd_comments(product_id, pages, comment_type):
    # 创建保存目录（如果不存在）
    save_dir = os.path.join(os.path.dirname(__file__), 'commentDate')
    os.makedirs(save_dir, exist_ok=True)  # 自动创建目录

    # 评价类型映射字典
    type_dict = {
        0: '全部评价',
        1: '追评',
        2: '好评',
        3: '中评',
        4: '差评'
    }
    type_name = type_dict[comment_type]

    # 构造完整文件路径
    file_path = os.path.join(save_dir, f'comments_{product_id}_{type_name}.csv')

    # with open(file_path, 'w', encoding='utf-8') as f:
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['用户', '评论'])  # 写入表头

        dp = ChromiumPage()

        try:
            # 访问商品详情页
            dp.get(f'https://item.jd.com/{product_id}.html')
            time.sleep(3)

            # 切换到商品评价标签
            tabs = dp.eles('tag:li@@data-tab=trigger')
            for tab in tabs:
                if '商品评价' in tab.text:
                    tab.click()
                    break
            time.sleep(3)

            # 切换到指定评价类型
            type_tabs = dp.eles('css:.filter-list li')
            target_tab = None
            for tab in type_tabs:
                if type_name in tab.text:
                    target_tab = tab
                    break

            if target_tab:
                target_tab.click()
                time.sleep(3)
            else:
                print(f"未找到{type_name}选项卡")
                return

            for current_page in tqdm(range(1, pages + 1), desc='爬取进度'):
                dp.scroll.to_bottom()
                time.sleep(2)

                comments = dp.eles('css:.comment-item')
                tqdm.write(f'第{current_page}页找到{len(comments)}条评论')

                for comment in comments:
                    try:
                        user = comment.ele('css:.user-info').text.replace('\n', ' ')
                        content = comment.ele('css:.comment-con').text.replace('\n', ' ')
                        writer.writerow([user, content])
                    except Exception as e:
                        tqdm.write(f'部分信息提取失败：{e}')

                # 翻页处理
                try:
                    next_btn = dp.ele('text:下一页')
                    if next_btn:
                        next_btn.click()
                        time.sleep(3)
                    else:
                        tqdm.write('已到最后一页')
                        break
                except Exception as e:
                    tqdm.write(f'翻页失败：{e}')
                    break


        finally:
            dp.quit()


if __name__ == '__main__':
    product_id = input("请输入商品ID（如100142621616）：").strip()
    pages = int(input("请输入要爬取的页数："))

    print("请选择评价类型：")
    print("0: 全部评价\n1: 追评\n2: 好评\n3: 中评\n4: 差评")
    comment_type = int(input("请输入数字0-4："))

    if comment_type not in range(5):
        print("输入错误，请重新运行程序并输入0-4之间的数字")
    else:
        get_jd_comments(product_id, pages, comment_type)
        print(f"评论已保存到 {os.path.join(os.getcwd(), 'commentDate')} 目录")