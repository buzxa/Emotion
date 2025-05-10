from DrissionPage import ChromiumPage
import time


def get_jd_comments(product_id, pages):
    # 创建txt文件
    with open(f'comments_{product_id}.txt', 'w', encoding='utf-8') as f:
        # 实例化浏览器对象（无头模式）
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

            current_page = 1
            while current_page <= pages:
                # 滚动加载全部内容
                dp.scroll.to_bottom()
                time.sleep(2)

                # 提取评论信息
                comments = dp.eles('css:.comment-item')
                print(f'第{current_page}页找到{len(comments)}条评论')

                # 遍历评论
                for comment in comments:
                    try:
                        # 用户名
                        user = comment.ele('css:.user-info').text.replace('\n', ' ')
                        # 评论内容（处理展开全文）
                        content = comment.ele('css:.comment-con').text.replace('\n', ' ')
                        f.write(f'用户：{user}\t评论：{content}\n')
                    except Exception as e:
                        print('部分信息提取失败：', e)

                # 翻页逻辑
                try:
                    next_btn = dp.ele('text:下一页')
                    if next_btn:
                        next_btn.click()
                        current_page += 1
                        time.sleep(3)
                    else:
                        print('已到最后一页')
                        break
                except Exception as e:
                    print('翻页失败：', e)
                    break

        finally:
            dp.quit()


if __name__ == '__main__':
    # 用户输入
    product_id = input("请输入商品ID：").strip()
    pages = int(input("请输入要爬取的页数："))

    # 执行爬取
    get_jd_comments(product_id, pages)
    print("评论爬取完成！")