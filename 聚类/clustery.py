from sklearn.cluster import KMeans
import numpy as np
import pandas as pd
import csv
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


def is_number(a):
    try:
        float(a)
        return True
    except ValueError:
        pass


def out_result(pre, uid2id):
    datas = []
    file_path = './weibo.csv'
    df = pd.read_excel(file_path, keep_default_na=False)
    cloumn = df.columns.tolist()
    cloumn.append('cluter_result')
    # print(cloumn)

    for index, row in df.iterrows():
        if (row['出度'] + row['入度'] <= 3):
            continue
        data = [row[i] for i in cloumn[0:-3]]
        data.append(pre[uid2id[row['uid']]])
        datas.append(data)
    cloumn = cloumn[0:-3] + [cloumn[-1]]
    print(cloumn)
    with open('./result_K=5.csv', 'w', encoding='utf_8_sig', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(cloumn)
        for data in datas:
            writer.writerow(data)


def reduce_demimension(features):
    ts = TSNE(n_components=2)
    ts.fit_transform(features)
    return ts.embedding_


if __name__ == '__main__':
    feature_file = './weibo.features'
    features = []
    K = 5
    uid2id = {}
    with open(feature_file, 'r') as f:
        for line in f:
            line = line.strip().split('\t')
            idx, uid = int(line[0]), int(line[1])
            uid2id[uid] = idx
            feature = eval(line[2])
            features.append(feature)
    features = np.array(features)
    fe_re_dimen = reduce_demimension(features)
    print(features.shape)
    model = KMeans(n_clusters=K)
    model.fit(features)
    pre = model.predict(features).tolist()
    # plt.scatter(fe_re_dimen[:, 0], fe_re_dimen[:, 1], marker='.', c=pre)
    # plt.show()
    out_result(pre, uid2id)
    type0_x = []  # 一共有3类，所以定义3个空列表准备接受数据
    type0_y = []
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    type4_x = []
    type4_y = []
    for i in range(len(pre)):
        if (pre[i] == 0):
            type0_x.append(fe_re_dimen[i][0])
            type0_y.append(fe_re_dimen[i][1])
        if (pre[i] == 1):
            type1_x.append(fe_re_dimen[i][0])
            type1_y.append(fe_re_dimen[i][1])
        if (pre[i] == 2):
            type2_x.append(fe_re_dimen[i][0])
            type2_y.append(fe_re_dimen[i][1])
        if (pre[i] == 3):
            type3_x.append(fe_re_dimen[i][0])
            type3_y.append(fe_re_dimen[i][1])
        if (pre[i] == 4):
            type4_x.append(fe_re_dimen[i][0])
            type4_y.append(fe_re_dimen[i][1])
    if (K == 3):
        plt.scatter(type0_x, type0_y, s=20, c='r', label='type0')
        plt.scatter(type1_x, type1_y, s=20, c='b', label='type1')
        plt.scatter(type2_x, type2_y, s=20, c='k', label='type2')
        plt.legend()
        plt.show()
    if (K == 4):
        plt.scatter(type0_x, type0_y, s=20, c='r', label='type0')
        plt.scatter(type1_x, type1_y, s=20, c='b', label='type1')
        plt.scatter(type2_x, type2_y, s=20, c='k', label='type2')
        plt.scatter(type3_x, type3_y, s=20, c='g', label='type3')
        plt.legend()
        plt.show()
    if (K == 5):
        plt.scatter(type0_x, type0_y, s=20, c='r', label='type0')
        plt.scatter(type1_x, type1_y, s=20, c='b', label='type1')
        plt.scatter(type2_x, type2_y, s=20, c='k', label='type2')
        plt.scatter(type3_x, type3_y, s=20, c='g', label='type3')
        plt.scatter(type4_x, type4_y, s=20, c='y', label='type4')
        plt.legend()
        plt.show()

