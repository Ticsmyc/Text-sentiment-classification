import requests
import re
import pymysql
import time


def get_comment_info(html):
    get_id = re.findall(r'<a.+people.+">(.+)</a>', html)
    get_stars = re.findall(r'<span class="allstar(.+) rating" title=".+"></span>', html)
    get_eval = re.findall(r'<span class=".+" title="(.+)"></span>', html)
    get_comments = re.findall(r'<p class="">(.+)', html)
    get_votes = re.findall(r'<span class="votes">(.+)</span>', html)
    get_date = re.findall(r'201[0-9]-[0-9]+-[0-9]+', html, re.S)
    n_page = re.findall(r'<a href="\?start=(\d+)&.+".+>', html)

    db = pymysql.connect("localhost", "root", "5307", "douban_data", charset='utf8')
    cursor = db.cursor()

    if get_id:
        for i in range(0, 19):
            sql = """INSERT INTO movies_comments (id, stars, eval, comment_info, votes, date)
                VALUES(%s,%s,%s,%s,%s,%s)"""
            try:
                cursor.execute(sql, (get_id[i], get_stars[i], get_eval[i], get_comments[i], get_votes[i], get_date[i]))
                db.commit()
            except Exception as e:
                db.rollback()
                continue
            i += 1

    db.close()
    return n_page


if __name__ == '__main__':
    # login douban.com
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/60.0.3112.113 Safari/537.36'
    }

    cookies = {
        'cookie': '使用你的cookies'
    }

    each = 25360
    while 1:
        url = 'https://movie.douban.com/subject/26608228/comments?start=' + str(
            each) + '&limit=20&sort=new_score&status=P'
        url_info = requests.get(url, cookies=cookies, headers=headers)
        print("正在抓取评论，从第" + str(each) + "条开始")
        next_page = get_comment_info(url_info.text)
        time.sleep(10)
        each = next_page[-1]
