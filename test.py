# 导入相关依赖库
import requests
import os

# 模拟请求头——反爬
head = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/113.0.0.0 Safari/537.36 Edg/113.0.1774.50'
}

s = 1

# 爬取10页
for page in range(30, 30 * 11, 30):
    print(f'pn={page}')
    # 请求的url
    url = f'https://image.baidu.com/search/acjson?tn=resultjson_com&logid=12061163953003980941&ipn=rj&ct=201326592&is=&fp=result&fr=&word=%E6%96%B0%E5%A8%98&queryWord=%E6%96%B0%E5%A8%98&cl=2&lm=-1&ie=utf-8&oe=utf-8&adpicid=&st=-1&z=&ic=0&hd=&latest=&copyright=&s=&se=&tab=&width=&height=&face=0&istype=2&qc=&nc=1&expermode=&nojc=&isAsync=&pn={page}&rn=30&gsm=1e&1684841207815='

    # 发送请求获取数据
    resp = requests.get(url=url, headers=head)

    # print(resp.text)

    # 转json，解析data所有数据
    data_json = resp.json()
    data = data_json['data']
    # print(data)

    print(len(data))
    # 遍历data下所有数据
    for i in data:
        try:
            # 解析图片url
            pic_url = i['thumbURL']
            print(pic_url)

            # # 请求图片url
            img = requests.get(pic_url, headers=head)
            # 保存图片
            # 设置保存的文件夹
            dir = 'ouputs'
            # 判断文件夹是否存在
            if not os.path.exists(dir):
                os.mkdir(dir)
            with open(f'{dir}/新娘_{s}.jpg', 'wb') as f:
                # 二进制的格式保存
                f.write(img.content)
                s += 1

        except:
            pass
