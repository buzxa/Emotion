from DrissionPage import ChromiumPage
import time
import os
import csv
from tqdm import tqdm

def get_jd_comments(product_id, pages, comment_type, sort_type, progress_callback=None):
    # save_dir = os.path.join(os.path.dirname(__file__), 'commentDate')
    save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'commentDate'))  # 改为绝对路径
    os.makedirs(save_dir, exist_ok=True)

    type_dict = {0: '全部评价', 1: '追评', 2: '好评', 3: '中评', 4: '差评'}
    type_name = type_dict[comment_type]
    file_path = os.path.join(save_dir,
                             f'comments_{product_id}_{type_name}_{"time" if sort_type == 1 else "default"}.csv')

    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['用户', '评论'])

        dp = ChromiumPage()
        existing_comments = set()

        try:
            dp.get(f'https://item.jd.com/{product_id}.html')
            time.sleep(3)

            # 切换到商品评价主标签页
            tab = dp.ele('text:商品评价', timeout=10)
            if tab:
                print("点击商品评价主标签")
                tab.click()
            else:
                print("未找到商品评价主标签")
                return
            time.sleep(3)

            # 切换评价类型
            type_tabs = dp.eles('css:.filter-list li')
            target_tab = None
            for tab in type_tabs:
                if type_name in tab.text:
                    print(f"找到评价类型: {tab.text}")
                    target_tab = tab
                    break
            if not target_tab:
                print(f"未找到{type_name}选项")
                return
            target_tab.click()
            time.sleep(2)

            # 切换排序方式
            sort_btn = dp.ele('css:.J-current-sortType', timeout=5)
            if sort_btn:
                sort_btn.click()
                time.sleep(1)
                sort_option = dp.ele(f'text:{"时间排序" if sort_type == 1 else "默认排序"}', timeout=5)
                if sort_option:
                    sort_option.click()
                    print("排序方式切换成功")
                    # 使用显式等待代替ele_loaded
                    start_time = time.time()
                    while time.time() - start_time < 10:
                        if dp.eles('css:.comment-item'):
                            break
                        time.sleep(0.5)
                    else:
                        print("等待评论加载超时")
                else:
                    print("未找到排序选项")
            else:
                print("无排序按钮")

            for current_page in tqdm(range(1, pages + 1), desc='爬取进度'):
                comments = []
                start_time = time.time()
                while time.time() - start_time < 30:
                    comments = dp.eles('css:.comment-item')
                    if comments:
                        tqdm.write(f'第{current_page}页找到{len(comments)}条评论')
                        break
                    time.sleep(2)
                else:
                    tqdm.write("等待评论超时")
                    break

                for comment in comments:
                    try:
                        user_elem = comment.ele('css:.user-info .name', timeout=1)
                        user = user_elem.text.strip() if user_elem else '匿名用户'

                        content_elem = comment.ele('css:.comment-con', timeout=1)
                        content = content_elem.text.strip() if content_elem else '无内容'

                        if (user, content) not in existing_comments:
                            writer.writerow([user, content])
                            existing_comments.add((user, content))
                    except Exception as e:
                        tqdm.write(f'提取失败: {e}')
                        continue


                # 翻页处理
                next_btn = dp.ele('text:下一页', timeout=2)
                if next_btn and 'disabled' not in next_btn.attr('class'):  # 检查是否可点击
                    next_btn.click()
                    time.sleep(1)
                else:
                    tqdm.write('已到最后一页')
                    break

                # 每处理完一页后触发回调
                if progress_callback:
                    progress_percent = int((current_page / pages) * 100)
                    progress_callback(progress_percent)

        finally:
            return file_path
            # dp.quit()

if __name__ == '__main__':
    product_id = input("请输入商品ID（如100142621616）：").strip()
    pages = int(input("请输入要爬取的页数："))

    print("请选择评价类型：")
    print("0: 全部评价\n1: 追评\n2: 好评\n3: 中评\n4: 差评")
    comment_type = int(input("请输入数字0-4："))

    if comment_type not in range(5):
        print("输入错误，请重新运行程序并输入0-4之间的数字")
    else:
        print("请选择排序方式：")
        print("0: 默认排序\n1: 时间排序")
        sort_type = int(input("请输入数字0或1："))

        if sort_type not in range(2):
            print("输入错误，请重新运行程序并输入0-1之间的数字")
        else:
            get_jd_comments(product_id, pages, comment_type, sort_type)
            print(f"评论已保存到 {os.path.join(os.getcwd(), 'commentDate')} 目录")