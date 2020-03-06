__author__ = 'gem'
import sys
import numpy as np  # 数据结构
import sklearn.cluster as skc  # 密度聚类
from sklearn import metrics  # 评估模型
# import matplotlib.pyplot as plt  # 可视化绘图
import pandas as pd
from itertools import cycle  ##python自带的迭代器模块
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.cluster import AgglomerativeClustering
import cx_Oracle as oracle

db = oracle.connect('ideal_mh/ideal_mh110@15.75.1.205:8080/minhang')
# db = oracle.connect('ideal_ja/ideal_ja110@192.168.0.122:1521/ja110')

cursor = db.cursor()


def get_db_data2(region):
    sql = "select AF_ADDR,to_number(AMAP_GPS_X) AMAP_GPS_X,to_number(AMAP_GPS_Y) AMAP_GPS_Y FROM region_spot where  amap_gps_x is not null and amap_gps_y is not null and amap_gps_x <> 0 and amap_gps_y <> 0  "

    if region != None and region != '':
        sql += " and region_name='" + region + "'"
    rs = cursor.execute(sql).fetchall()
    data = pd.DataFrame(rs, columns=["AF_ADDR", 'AMAP_GPS_X', 'AMAP_GPS_Y'])
    return data


def get_db_data(startDate, endDate, startTime, endTime, region, ay):
    sql = "select nvl(spot_addr,AF_ADDR) AF_ADDR,to_number(AMAP_GPS_X) AMAP_GPS_X,to_number(AMAP_GPS_Y) AMAP_GPS_Y from pd_police_all_data where  amap_gps_x is not null and amap_gps_y is not null and amap_gps_x <> 0 and amap_gps_y <> 0  "
    if ay != None and ay != '':
        if ay == '报警类案件':
            sql += " and bjay_type='报警类案件' "
        if ay == '黄赌毒打架':
            sql += " and (bjay1='黄赌毒' or bjay2='打架斗殴') "

    if endDate != None and endDate != '':
        sql += " and date1>=add_months(to_date('" + endDate + "','yyyyMMdd'),-1)"
        sql += " and date1<=to_date('" + endDate + "','yyyyMMdd')"

    if startTime != None and startTime != '' and endTime != None and endTime != '':
        if int(startTime) > int(endTime):
            sql += " and (substr(jjd_id,9,6) >='" + startTime + "' and substr(jjd_id,9,6) <'240000'"
            sql += " or substr(jjd_id,9,6) >='000000' and substr(jjd_id,9,6)  <'" + endTime + "')"
        else:
            sql += " and substr(jjd_id,9,6) >='" + startTime + "'"
            sql += " and substr(jjd_id,9,6) <'" + endTime + "'"

    if region != None and region != '':
        sql += " and regionname='" + region + "'"
    rs = cursor.execute(sql).fetchall()
    data = pd.DataFrame(rs, columns=["AF_ADDR", 'AMAP_GPS_X', 'AMAP_GPS_Y'])
    return data


def get_csv_data():
    file = pd.read_excel("bbb.xls")
    df2 = pd.DataFrame(file)
    return df2.iloc[:, 0:3]


def print_eval_dbscan(db, X):
    labels = db.labels_  # 和X同一个维度，labels对应索引序号的值 为她所在簇的序号。若簇编号为-1，表示为噪声

    raito = len(labels[labels[:] == -1]) / len(labels)  # 计算噪声点个数占总数的比例
    print('噪声比:', format(raito, '.2%'))

    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)  # 获取分簇的数目

    print('分簇的数目: %d' % n_clusters_)
    # print("轮廓系数: %0.3f" % metrics.silhouette_score(X, labels))  # 轮廓系数评价聚类的好坏

    # 绘图

    for i in range(n_clusters_):
        print('簇 ', i, '的所有样本:')
        one_cluster = X[labels == i]
        print(len(one_cluster))

        #  print(one_cluster)
        # plt.plot(one_cluster[:, 0], one_cluster[:, 1], 'o')

    # plt.show()
    return labels, n_clusters_


def agglomerative_cluster(X, cluster_low=1, cluster_high=10):
    n_cluster = 3
    silhouette_score = 0
    fit_cluster = n_cluster
    if (len(X) > 3):
        while n_cluster <= 10 and n_cluster >= 3 and n_cluster < len(X):
            clustering = AgglomerativeClustering(n_clusters=n_cluster).fit(X)
            try:
                labels_x = clustering.labels_
                silhouette_score_tmp = metrics.silhouette_score(X, labels_x)
                if silhouette_score_tmp > silhouette_score:
                    silhouette_score = silhouette_score_tmp
                    fit_cluster = n_cluster
                    print("轮廓系数: %0.3f" % silhouette_score)
            except:
                print("轮廓系数異常")
            n_cluster = n_cluster + 1
    else:
        n_clusters_1 = 1
        fit_cluster = n_clusters_1
        while n_clusters_1 >= 1 and n_clusters_1 <= len(X):
            clustering = AgglomerativeClustering(n_clusters=n_clusters_1).fit(X)
            try:
                labels_x = clustering.labels_
                silhouette_score_tmp = metrics.silhouette_score(X, labels_x)
                if silhouette_score_tmp > silhouette_score:
                    silhouette_score = silhouette_score_tmp
                    fit_cluster = n_cluster
                    print("轮廓系数: %0.3f" % silhouette_score)
            except:
                print("轮廓系数異常")
            n_clusters_1 = n_clusters_1 + 1
    print("最优分类数：%0.3f" % fit_cluster)
    clustering = AgglomerativeClustering(n_clusters=fit_cluster).fit(X)

    labels = clustering.labels_
    n_clusters = len(np.unique(labels))

    print(n_clusters)
    # print("轮廓系数: %0.3f" % metrics.silhouette_score(X, labels))

    for i in range(n_clusters):
        print('簇 ', i, '的所有样本:')
        one_cluster = X[labels == i]
        print(len(one_cluster))

        #  print(one_cluster)

        # plt.plot(one_cluster[:, 0], one_cluster[:, 1], 'o')
    # plt.title('agglomerativeclustering %d' % n_clusters)

    # plt.show()
    return labels, n_clusters


def mean_shift(xx, span_low=0.2, span_high=0.7):
    result = []
    span = 0.7
    res_dict = {}
    silhouette_score = 0
    fit_span = span
    has_result = False
    while span <= span_high and span >= span_low:
        bandwidth = estimate_bandwidth(xx, quantile=span)
        if bandwidth != 0:
            has_result = True
            ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
            ms.fit(xx)
            labels_x = ms.labels_
            try:
                silhouette_score_tmp = metrics.silhouette_score(xx, labels_x)
                if silhouette_score_tmp > silhouette_score:
                    silhouette_score = silhouette_score_tmp
                    fit_span = span
                print("轮廓系数: %0.3f" % metrics.silhouette_score(xx, labels_x))  # 轮廓系数评价聚类的好坏
            except ValueError as e:
                print("轮廓系数異常")

        span = span - 0.1

    print(fit_span)
    print("轮廓系数最佳:: %0.3f" % silhouette_score)
    if has_result == True:
        bandwidth = estimate_bandwidth(xx, quantile=fit_span)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(xx)
        labels_x = ms.labels_
        # print("轮廓系数: %0.3f" % metrics.silhouette_score(xx, labels_x)) #轮廓系数评价聚类的好坏
        cluster_centers = ms.cluster_centers_
        ##总共的标签分类
        labels_unique = np.unique(labels_x)
        ##聚簇的个数，即分类的个数
        n_clusters_x = len(labels_unique)

        print("number of estimated clusters : %d" % n_clusters_x)

        for k in zip(range(n_clusters_x)):
            my_members = labels_x == k
            cluster_center = cluster_centers[k]
            print(cluster_center)
            xx = np.array(xx)
            result.append({"AMAP_GPS_X": cluster_center[0], "AMAP_GPS_Y": cluster_center[1], "CN": len(xx[my_members])})
            # #绘图
        # plt.figure(1)
        # plt.clf()
        colors = cycle('bgrcmykbgrcmykbgrcmykbgrcmyk')
        for k, col in zip(range(n_clusters_x), colors):
            ##根据lables中的值是否等于k，重新组成一个True、False的数组
            my_members = labels_x == k
            cluster_center = cluster_centers[k]
            print(k)
            print(cluster_center)
            ##X[my_members, 0] 取出my_members对应位置为True的值的横坐标
            xx = np.array(xx)
            # plt.plot(xx[my_members, 0], xx[my_members, 1], col + '.')
            # plt.plot(cluster_center[0], cluster_center[1], 'o', markerfacecolor=col,
            #          markeredgecolor='k', markersize=14)
        # plt.title('Estimated number of clusters: %d' % n_clusters_x)
        # plt.show()
    return result, has_result


def cluster(startDate, endDate, startTime, endTime, region, ay):
    # data = get_csv_data()

    # data_orgin = get_db_data2( region)

    data_orgin = get_db_data(startDate, endDate, startTime, endTime, region, ay)

    if data_orgin.shape[0] == 0:
        return

    data = data_orgin[['AMAP_GPS_X', 'AMAP_GPS_Y']]
    X = np.array(data)

    db = skc.DBSCAN(eps=0.005, min_samples=1).fit(X)  # DBSCAN聚类方法 还有参数，matric = ""距离计算方法

    labels, n_clusters_ = print_eval_dbscan(db, X)

    result = []

    res_dic = {}
    X_remains = []

    for i in range(n_clusters_):
        print('簇 ', i, '的所有样本:')
        one_cluster = X[labels == i]
        xx = np.array(one_cluster)

        span_low = 0.2
        if len(xx) < 10:
            for c in one_cluster:
                X_remains.append(c)
        else:
            result_tmp, has_result = mean_shift(xx, span_low=span_low)
            if has_result == True:
                for r in result_tmp:
                    result.append(r)
            else:
                for c in one_cluster:
                    X_remains.append(c)

    if len(X_remains) >= 10:
        result_remain, has_result = mean_shift(np.array(X_remains))
        if has_result == True:
            for r in result_remain:
                result.append(r)

    # file=pd.read_excel("bbb.xls")
    # df2 = pd.DataFrame(file)
    df2 = data_orgin

    if len(result) >= (n_clusters_ - 1):
        all_coords = np.array(df2[["AMAP_GPS_X", "AMAP_GPS_Y"]])

        result_final = []
        for r in result:
            center = np.array([r["AMAP_GPS_X"], r["AMAP_GPS_Y"]])
            d = center - all_coords
            dis1 = np.square(d)
            df3 = pd.DataFrame(dis1)
            dis = np.array(df3.iloc[:, 0:1]) + np.array(df3.iloc[:, 1:2])
            dis = np.sqrt(dis)
            print(np.min(dis))
            p, q = pd.DataFrame(dis).stack().idxmin()
            print(p)
            print(r["CN"])
            print(df2.iloc[p])

            result_final.append([df2.ix[p]["AF_ADDR"], df2.ix[p]["AMAP_GPS_X"], df2.ix[p]["AMAP_GPS_Y"], r["CN"]])
    else:
        result_final = []
        A_X = np.array(data_orgin)
        agg_labels, n_agg_clusters = agglomerative_cluster(X)
        for i in range(n_agg_clusters):
            print('簇 ', i, '的所有样本:')
            one_cluster = A_X[agg_labels == i]
            xx = np.array(one_cluster)
            min_dist = 1
            center_point = None
            for j in xx:
                distance = cost(j[1:3], xx[:, 1:3])
                print(distance)
                if (distance < min_dist):
                    min_dist = distance
                    center_point = j
            result_final.append([center_point[0], center_point[1], center_point[2], len(xx)])
            print(center_point)

    # df = pd.DataFrame(np.array(result_final), columns=["AF_ADDR", "AMAP_GPS_X", "AMAP_GPS_Y", "CN"])
    # df.to_csv("data_fianl.csv", header=None, index=None)

    save_to_db(startDate, endDate, startTime, endTime, region, result_final)


#
def save_to_db(startDate, endDate, startTime, endTime, region, result):
    for res in result:
        xzb = res[1]
        yzb = res[2]
        if xzb.startswith('31'):
            xzb = res[2]
            yzb = res[1]

        cursor.execute(
            'insert into PD_POLICE_CLUSTER_PEEK(id,start_Date, end_Date, start_Time, end_Time, region,amap_gps_x,amap_gps_y,af_addr,cn) values '
            '(seq_CLUSTER_PEEK.nextval,:1, :2, :3, :4, :5,:6,:7,:8,:9)',
            [startDate, endDate, startTime, endTime, region, xzb, yzb, res[0], res[3]])
        db.commit()


def cost(c, all_points):  # 指定点，all_points:为集合类的所有点
    return np.sum(np.sum((c - all_points) ** 2, axis=1) ** 0.5)


if __name__ == "__main__":
    argv1 = sys.argv[1]
    if argv1 == None or argv1 == '':
        exit()
    startDate = "20190613"
    endDate = argv1
    startTime = "083000"
    endTime = "113000"
    region = "碧江第一责任区"
    ay = '报警类案件'
    peroidArr = [("083000", "113000", "报警类案件"), ("113000", "133000", "报警类案件"), ("133000", "163000", "报警类案件"),
                 ("183000", "230000", "黄赌毒打架"), ("230000", "063000", "报警类案件")]
    cursor.execute('delete from PD_POLICE_CLUSTER_PEEK where end_Date=:1', [endDate])
    for peroid in peroidArr:
        sql = "select distinct regionname_alias from data_shanghairegioninfo_new where type=3 and regionname_alias is not null"
        rs = cursor.execute(sql).fetchall()
        for row in rs:
            region = row[0]
            print(region)
            cluster('', endDate, peroid[0], peroid[1], region, peroid[2])

    # cluster(startDate, endDate, startTime, endTime, region,ay)
    cursor.close()
    db.close()
