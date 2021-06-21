import numpy as np
import csv
import pandas as pd
from collections import Counter

type2feats = {}
if __name__ == '__main__':
    K = 5
    file = './result_K=5.csv'
    df = pd.read_csv(file, keep_default_na=False, encoding='gbk')
    # 出入度和的平均值、粉丝数的平均值：衡量其知名度
    # 微博数：衡量其活跃度
    # 领域
    # 是否认证
    # 是否官方：认证原因
    # 标签
    type2feats[0] = []
    type2feats[1] = []
    type2feats[2] = []
    type2feats[3] = []
    type2feats[4] = []
    column2index = {'uname': 0, '出度': 1, '入度': 2}
    for index, row in df.iterrows():
        uid = row['uid']
        u_name = row['uname']
        out_degree = int(row['出度'])
        in_degree = int(row['入度'])
        area = row['所涉领域']
        attest = row['是否认证']
        follow_num = float(row['关注数']) if row['关注数'] != '' else 0
        follower_num = float(row['粉丝数']) if row['粉丝数'] != '' else 0
        weibo_num = float(row['微博数']) if row['微博数'] != '' else 0

        attest_reason = str(row['认证原因'])
        intro = str(row['简介'])
        label = str(row['标签'])
        address = str(row['地址']) if row['地址'] != '' else '其他'

        gender = 0 if row['性别'] == '女' else 1
        # birthday = row['生日']
        # print(row['注册时间'])
        sign_up_time = row['注册时间'] if row['注册时间'] != '' else 0
        cluter_result = row['cluter_result']
        feat = [uid, u_name, out_degree, in_degree, area, attest, attest_reason, follow_num, follower_num, weibo_num,
                intro, label, address, gender, sign_up_time, cluter_result]
        type2feats[cluter_result].append(feat)
    for i in range(K):
        num = len(type2feats[i])
        type2feats[i] = np.array(type2feats[i])
        print('类别：{},人数{}'.format(i, num))
        print('出度平均值:{}'.format(np.mean(type2feats[i][:, 2].astype(int))))
        print('入度平均值:{}'.format(np.mean(type2feats[i][:, 3].astype(int))))
        print('粉丝数平均值:{}'.format(np.mean(type2feats[i][:, 8].astype(float))))
        print('微博数平均值:{}'.format(np.mean(type2feats[i][:, 9].astype(float))))
        attest_num = Counter(type2feats[i][:, 5])['True']
        not_attest_num = Counter(type2feats[i][:, 5])['False']
        rate_t = attest_num / num
        rate_f = not_attest_num / num
        print('认证情况:{} {} {}'.format(Counter(type2feats[i][:, 5]), rate_t, rate_f))

        area = Counter(type2feats[i][:, 4])
        for area1, count in area.items():
            area[area1] /= num
        print(area)

        gender_rate = Counter(type2feats[i][:, -3])
        for gender1, count in gender_rate.items():
            gender_rate[gender1] /= num
        print(gender_rate)
