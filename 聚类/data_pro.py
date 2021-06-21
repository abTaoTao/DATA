import pandas as pd
from sklearn.preprocessing import StandardScaler
import math
import numpy as np
import text2vec
import datetime
import matplotlib.pyplot as plt

def data_read():
    file_path = './weibo.csv'
    df = pd.read_excel(file_path, keep_default_na=False)
    user2feats = {}
    uid2id = {}
    id2uid = {}
    id_ = 0
    max_area_num = 0
    for index, row in df.iterrows():
        # print(row)
        uid = row['uid']
        u_name = row['uname']
        out_degree = int(row['出度'])
        in_degree = int(row['入度'])
        if (out_degree + in_degree <= 3):
            continue
        out_degree = math.log(out_degree + 1)
        in_degree = math.log(in_degree + 1)
        area = [0, 0, 0, 0, 0]
        area[row['所涉领域'] - 1] = 1
        attest = [1] if row['是否认证'] == True else [0]

        # follow_num = float(row['关注数']) if row['关注数'] != '' else 0
        # follower_num = float(row['粉丝数']) if row['粉丝数'] != '' else 0
        # weibo_num = float(row['微博数']) if row['微博数'] != '' else 0
        follow_num = math.log(float(row['关注数']) + 1) if row['关注数'] != '' else 0
        follower_num = math.log(float(row['粉丝数']) + 1) if row['粉丝数'] != '' else 0
        weibo_num = math.log(float(row['微博数']) + 1) if row['微博数'] != '' else 0

        attest_reason = str(row['认证原因'])
        intro = str(row['简介'])
        label = str(row['标签'])
        address = str(row['地址']) if row['地址'] != '' else '其他'

        gender = [0] if row['性别'] == '女' else [1]
        # birthday = row['生日']
        # print(row['注册时间'])
        sign_up_time = math.log(row['注册时间'] + 1) if row['注册时间'] != '' else 0

        feats = [uid, u_name, out_degree, in_degree, area, attest, attest_reason, follow_num, follower_num, weibo_num,
                 intro, label, address, gender, sign_up_time]

        if (uid in user2feats):
            user2feats[uid][4] = [1, 0, 1, 0, 0]
            print(feats)
        else:
            user2feats[uid] = feats

        if (uid not in uid2id):
            uid2id[uid] = id_
            id2uid[id_] = uid
            id_ += 1

    return user2feats, uid2id, id2uid


def get_text_emb(text):
    if text[0] == '' or type(text[0]) != str:
        return np.array([0 for i in range(96)])
    t2v = text2vec.text2vec(text)
    emb = t2v.avg_wv()
    return emb


def data_feature_pro(user2feats, uid2id, id2uid):
    features = []
    out_degrees = []
    in_degrees = []
    follow_nums = []
    follower_nums = []
    weibo_nums = []
    sign_up_times = []
    for usid, feats in user2feats.items():
        out_degrees.append(feats[2])
        in_degrees.append(feats[3])
        follow_nums.append(feats[7])
        follower_nums.append(feats[8])
        weibo_nums.append(feats[9])
        sign_up_times.append(feats[-1])
    std = StandardScaler()

    out_degrees = std.fit_transform(np.array(out_degrees).reshape(-1, 1)).tolist()
    in_degrees = std.fit_transform(np.array(in_degrees).reshape(-1, 1)).tolist()
    follow_nums = std.fit_transform(np.array(follow_nums).reshape(-1, 1)).tolist()
    follower_nums = std.fit_transform(np.array(follower_nums).reshape(-1, 1)).tolist()
    weibo_nums = std.fit_transform(np.array(weibo_nums).reshape(-1, 1)).tolist()
    sign_up_times = std.fit_transform(np.array(sign_up_times).reshape(-1, 1)).tolist()
    idx = 0
    with open('weibo.features', 'w') as f:
        for usid, feats in user2feats.items():
            feats[2], feats[3], feats[7], feats[8], feats[9], feats[-1] = out_degrees[idx], in_degrees[idx], \
                                                                          follow_nums[
                                                                              idx], follower_nums[idx], weibo_nums[idx], \
                                                                          sign_up_times[idx]
            # print(feats[-1])
            # break
            feats[6] = get_text_emb([feats[6]]).reshape(96).tolist()
            feats[10] = get_text_emb([feats[10]]).reshape(96).tolist()
            feats[11] = get_text_emb([feats[11]]).reshape(96).tolist()
            feats[12] = get_text_emb([feats[12]]).reshape(96).tolist()

            feature = []
            # print(feats)
            for i in range(2, len(feats)):
                feature.extend(feats[i])
            # print(len(feature))
            features.append(feature)
            f.write('{}\t{}\t{}\n'.format(idx, id2uid[idx], feature))
            if (idx % 10 == 0):
                print('has finished {}/{} user emb time : {}'.format(idx, len(id2uid), datetime.datetime.now()))
            idx += 1
    return features


if __name__ == '__main__':
    user2feats, uid2id, id2uid = data_read()
    uid_num = len(user2feats)
    print(len(user2feats), len(uid2id))
    # features = data_feature_pro(user2feats, uid2id, id2uid)
