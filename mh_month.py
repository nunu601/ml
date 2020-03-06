import pandas as pd
import numpy as np
import calendar
from fbprophet import Prophet
import cx_Oracle as oracle
# df = pd.read_csv('example_wp_log_peyton_manning.csv')

db_url = 'ideal_mh/ideal_mh110@192.168.0.122:1521/mh110'

def appendSqlByAy(ay):
    sql = ""
    if ay=='全部':
        None
    elif ay=='报警类案件':
        sql+= " and bjay_type = '报警类案件'"
    elif ay == '黄赌':
        sql+= " and bjay2 in ('赌博类','色情类')"
    elif ay == '色情类':
        sql += " and bjay2 in ('色情类')"
    elif ay == '赌博类':
        sql += " and bjay2 in ('赌博类')"
    elif ay == '侵犯人身权利':
        sql += " and bjay1 in ('侵犯人身权利')"
    return sql


def pred(ay, pcs):
    db = oracle.connect(db_url)

    cursor = db.cursor()

    sql = "select to_char(date1,'yyyy-MM-dd') ds,count(*) y " \
          "from pd_police_all_data where date1>=to_date('2016-01-01','yyyy-MM-dd') " \


    sql+=appendSqlByAy(ay)

    if pcs != None and pcs != '' and pcs !='分局':
        sql += " and cjdw='%s' " % (pcs)

    sql += " group by date1 order by date1"
    print(sql)
    rs = cursor.execute(sql).fetchall()
    if len(rs) == 0:
        cursor.close()
        db.close()
        return  pd.DataFrame()
    data = pd.DataFrame(rs, columns=['ds', 'y'])
    cursor.close()
    db.close()

    return data

# df = pred(pcs=None,ay=None)
# # df = pd.read_csv('mh_jj.csv')
# df['y'] = np.log(df['y'])
#
# print(df.head())


chunjie = pd.DataFrame({
  'holiday': 'chunjie',
  'ds': pd.to_datetime(['2016-02-07', '2016-02-08', '2016-02-09', '2016-02-10', '2016-02-11', '2016-02-12', '2016-02-13',
                        '2017-01-27', '2017-01-28', '2017-01-29', '2017-01-30', '2017-01-31', '2017-02-01', '2017-02-02',
                        '2018-02-15', '2018-02-16', '2018-02-17', '2018-02-18', '2018-02-19', '2018-02-20', '2018-02-21',
                        '2019-02-04', '2019-02-05', '2019-02-06', '2019-02-07', '2019-02-08', '2019-02-09', '2019-02-10',
                        '2020-01-24', '2020-01-25', '2020-01-26', '2020-01-27', '2020-01-28', '2020-01-29', '2020-01-30']),
  'lower_window': 0,
  'upper_window': 1,
})

khn = pd.DataFrame({
  'holiday': 'khn',
  'ds': pd.to_datetime(['2016-12-01','2017-12-01','2018-12-01','2019-12-01']),
  'lower_window': 0,
  'upper_window': 1,
})



holidays = pd.concat((chunjie, khn))
import time

def doForecast(forecastDate,month_days,ay,pcs):
    v_pcs = pcs
    if pcs == '':
        v_pcs='分局'


    in_date_str = forecastDate




    df = pred(pcs=v_pcs, ay=ay)
    # df = pd.read_csv('mh_jj.csv')
    if len(df) >= 2:

        df['y'] = np.log(df['y'])

        df = df.loc[(df['ds'] < forecastDate)]
        # m = Prophet(holidays=chunjie, changepoints=changepoints)
        m = Prophet(holidays=chunjie)
        m.fit(df)
        future = m.make_future_dataframe(periods=month_days)
        # print(future.tail())
        forecast = m.predict(future)
        forecast['yhat'] = np.exp(forecast['yhat'])
        df2 = pd.DataFrame(forecast['yhat'].tail(month_days))
        print(df2.head())
        print(df2.apply(sum))
        month_pred = df2.apply(sum)
        in_time = time.mktime(time.strptime(in_date_str, '%Y-%m-%d'))
        lt = time.localtime(in_time)

        v_month = int(in_date_str[5:7])
        # cursor.execute(
        #     "insert into pd_police_yuce_num(pcs, year, month, yuce_num,ay,yuce_d,publish_flag) values "
        #     "(:1, to_char(to_date(:2,'yyyy-MM-dd') ,'yyyy'), :3, :4, :5,:6,1)",   [v_pcs, in_date_str, v_month, round(month_pred.values[0]), ay,yuce_d])
        for i in range(len(df2)):
            # cursor.execute(
            #     "insert into PD_POLICE_YUCE_NUM_DAY(pcs, year, day, yuce_num,ay,date1,yuce_d,publish_flag) "
            #     "values (:1, to_char(to_date(:2,'yyyy-MM-dd')+" + str(i) + ",'yyyy'), to_date(:2,'yyyy-MM-dd')+" + str(i) + "-trunc(to_date(:2,'yyyy-MM-dd')+" + str(i) + ",'yyyy')+1, :3, :4,to_date(:2,'yyyy-MM-dd')+" + str(i) + ",:5,1)",
            #     [v_pcs, in_date_str,in_date_str,in_date_str, round(df2.iloc[i].values[0]), ay,in_date_str, yuce_d])
            cursor.execute(
                "insert into PD_POLICE_YUCE_NUM_BASE(pcs, year, day, yuce_num,ay,date1) "
                "values (:1, to_char(to_date(:2,'yyyy-MM-dd')+" + str(i) + ",'yyyy'), to_date(:2,'yyyy-MM-dd')+" + str(i) + "-trunc(to_date(:2,'yyyy-MM-dd')+" + str(i) + ",'yyyy')+1, :3, :4,to_date(:2,'yyyy-MM-dd')+" + str(i) + ")",
                [v_pcs, in_date_str,in_date_str,in_date_str, round(df2.iloc[i].values[0]), ay,in_date_str])
        db.commit()

    # with open('result.txt', 'a+') as f:
    #     f.write(forecastDate+"----"+str(df2.apply(sum))+"\r\n")
    # m.plot(forecast).show()
    # m.plot_components(forecast).show()#绘制成分趋势图

date_arr=['2019-08-01']


import sys


for forecastDate in date_arr:
    monthRange = calendar.monthrange(int(forecastDate.split("-")[0]), int(forecastDate.split("-")[1]))
    # doForecast(df,forecastDate,monthRange[1])
qy = "派出所"
argv1 = sys.argv[1]
if argv1 != None and argv1 != '':

    db = oracle.connect(db_url)
    cursor = db.cursor()
    monthRange = calendar.monthrange(int(argv1.split("-")[0]), int(argv1.split("-")[1]))
    ayList = ['全部', '报警类案件']


    cursor.execute(
            'delete from PD_POLICE_YUCE_NUM_BASE where date1>=to_date(:1,:2) ' , [argv1, 'yyyy-MM-dd'])
    db.commit()


    v_year = argv1.split("-")[0]
    v_month = int(argv1.split("-")[1])

    for ay in ayList:
        doForecast( argv1, monthRange[1], ay, '分局')

    sql = "select distinct pcs from pd_police_pcs where qy=:1 and pcs not in('龙柏新村派出所','水上派出所','华银派出所')"
    rs = cursor.execute(sql, [qy]).fetchall()
    for row in rs:
        pcs = row[0]
        print(pcs)
        if pcs == None or pcs == '':
            continue
        for ay in ayList:
            doForecast(argv1, monthRange[1], ay, pcs)


    cursor.execute(
            "delete from PD_POLICE_YUCE_NUM_DAY where date1>=to_date(:1,:2) and ay in ('全部','报警类案件') " , [argv1, 'yyyy-MM-dd'])

    db.commit()
    # cursor.execute(
    #         "delete from PD_POLICE_YUCE_NUM where year=to_number(to_char(to_date(:1,'yyyy-MM-dd'),'yyyy')) and month = to_number(to_char(to_date(:2,'yyyy-MM-dd'),'MM'))  and ay in ('全部','报警类案件')" , [argv1,argv1])
    # db.commit()

    cursor.execute(
        "insert into PD_POLICE_YUCE_NUM_DAY(pcs, year, day, yuce_num,ay,date1,yuce_d,publish_flag) select a.PCS,a.Year,a.day,a.yuce_num,a.ay,a.date1,1,1 from PD_POLICE_YUCE_NUM_BASE a where date1>=to_date(:1,'yyyy-MM-dd')",
        [argv1])
    db.commit()

    # cursor.execute(
    #     "insert into PD_POLICE_YUCE_NUM(pcs,year,month,ay,Yuce_Num,publish_flag,yuce_d)"
    #     "  select pcs,:1,:2,ay,yuce_num,1,1 from (select sum(yuce_num) yuce_num,pcs,ay from  PD_POLICE_YUCE_NUM_BASE "
    #     "where to_char(date1,'yyyy') = :3 and to_char(date1,'MM') = :4 group by pcs, ay )",
    #     [v_year,str(v_month),v_year,str(v_month)])
    # db.commit()


    cursor.close()
    db.close()

