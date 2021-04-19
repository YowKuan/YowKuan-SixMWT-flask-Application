import os
import csv
import pandas as pd
from app.model.Speed_chart import Speed_chart
from app.model.Distance_bar import Distance_bar
from app.model.Distance_speed import Distance_speed
from app.model.Rounds import Rounds
from app.model.Video import Video
from app import db
from sqlalchemy import create_engine

engine = create_engine('postgresql://postgres:Csie346grad@localhost:5432/postgres')

def speed_dis_calc(speed_record, speed_record_time, Group, ID, Date):    
    #In this function, we calculate total walking distance and time-speed chart
    speed_list=[]
    speed_dict={}
    time_dict = {}
    dis_dict = {}
    #speed_record_error processing
    speed_mean = round(sum(speed_record)/len(speed_record), 1)
    for i in range(len(speed_record)):
        if speed_record[i] <= 0.25:
            speed_record[i] = speed_mean

    for i in range(0, len(speed_record)):
        speed_dict = {"Group":Group, "Date":Date, "ID":ID, "Time(s)":speed_record_time[i], "Speed(m/s)":speed_record[i] }
        speed_list.append(speed_dict)

    #calculate walking distance
    final_walking_distance = round(sum(speed_record) * 5)
    print("final_walking_dis", final_walking_distance)
    dis_dict = {"Group":Group, "Date":Date, "ID":ID, "Walking Distance(m)":final_walking_distance}    
    #Insert into database
    for i in speed_list:
        speed_insert = Speed_chart(patient_id = i['ID'],
            group = i['Group'],
            date = i['Date'],
            time = i['Time(s)'],
            speed = i['Speed(m/s)']
            )
        db.session.add(speed_insert)
    distance_insert = Distance_bar(patient_id = dis_dict['ID'],
            group = dis_dict['Group'],
            date = dis_dict['Date'],
            distance = dis_dict['Walking Distance(m)']
    )
    db.session.add(distance_insert)
    db.session.commit()

    speed_group_query = db.session.query(Speed_chart.group.distinct().label("group_number"))
    group_list = [a.group_number for a in speed_group_query.all()]
    print("group_list:", group_list)
    df = pd.read_sql_query('select * from "speed_chart"',con=engine)

    #Calculate average
    for i in group_list:
        for j in range(5, 365, 5):
            filter_func = Speed_chart.query.filter_by(patient_id = "average of group"+str(i), group = i, time= j).first()
            data = df[(df['group']==i) & (df['time']==j)]
            speed_data = round(data['speed'].mean(), 2)
            if filter_func:
                filter_func.speed = speed_data
                db.session.commit()
            else:
                speed_average = Speed_chart(patient_id = "average of group"+str(i), group=i, date = "2000-01-01", time=j, speed=speed_data)
                db.session.add(speed_average)
                db.session.commit()
    df = pd.read_sql_query('select * from "distance_bar"',con=engine)
    for i in group_list:
        filter_func = Distance_bar.query.filter_by(patient_id="average of group"+str(i), group = i).first()
        data = df[df['group']==i]
        distance_data = round(data['distance'].mean(), 2)
        if filter_func:
            filter_func.distance = distance_data
            db.session.commit()
        else:
            distance_average = Distance_bar(patient_id = "average of group"+str(i), group = i, date = "2000-01-01", distance=distance_data)
            db.session.add(distance_average)
    db.session.commit()

    return speed_list, dis_dict

def every_three_calculation(time_3M, Group, ID, Date):
    #In this function, we calculate total walking distance and distance-speed chart
    distance_speed_list = []
    avg_speed_list = []
    mov_avg = []
    for i in range(1, len(time_3M)):
        avg_speed = round(3/(time_3M[i]-time_3M[i-1]), 1)
        #convert uncommon speed into 1
        if avg_speed > 5:
            avg_speed = 1
        avg_speed_list.append(avg_speed)

        #ten points moving average
        if len(avg_speed_list)>=10:
            mov_avg.append(sum(avg_speed_list[-10:])/10)
        dis_speed_dict = {"Group":Group, "Date":Date, "ID":ID, "distance":i*3, "Dis_time(m/s)":avg_speed, "Moving_avg":None if len(avg_speed_list)<10 else mov_avg[-1], "time":round(time_3M[i])}
        distance_speed_list.append(dis_speed_dict)

    for i in distance_speed_list:
        distance_speed_insert = Distance_speed(patient_id = i['ID'],
            group = i['Group'],
            date = i['Date'],
            distance = i['distance'],
            speed = i['Dis_time(m/s)'],
            moving_avg= i['Moving_avg'],
            time = i['time']
            )
        db.session.add(distance_speed_insert)
    db.session.commit()

    dis_speed_group_query = db.session.query(Distance_speed.group.distinct().label("group_number"))
    group_list = [a.group_number for a in dis_speed_group_query.all()]
    df = pd.read_sql_query('select * from "distance_speed"',con=engine)
    max_dis_group = []
    mdg_index = 0
    for i in group_list:
        dff = df[df['group']==i]
        max_dis_group.append(int(dff['distance'].max()))
    print("max dis group", max_dis_group)
    for i in group_list:
        for j in range(3, max_dis_group[mdg_index], 3):
            filter_func = Distance_speed.query.filter_by(patient_id = "average of group"+str(i), group = i, distance= j).first()
            data = df[(df['group']==i) & (df['distance']==j)]
            speed_data = round(data['speed'].mean(), 2)
            time_data = round(data['time'].mean())
            moving_avg_data = round(data['moving_avg'].mean(), 2)
            if filter_func:
                filter_func.speed = speed_data
                filter_func.moving_avg = moving_avg_data
                db.session.commit()
            else:
                speed_average = Distance_speed(patient_id = "average of group"+str(i), group=i, date = "2000-01-01", distance=j, speed=speed_data, time= time_data, moving_avg=moving_avg_data)
                db.session.add(speed_average)
                db.session.commit()
        mdg_index+=1



def round_calculation(speed_record, Group, ID, Date):
    round_prev = 0
    round_over = 0
    k = 0
    round_record = []
    speed_record_for_round = speed_record
    while sum(speed_record_for_round)>=6:
        if round_over >= 30/5:
            #超過的時候，先把沒超過的部分取出來當prev
            round_prev = round(sum(speed_record_for_round[:k-1]), 2)
            #print("round:prev:", round_prev)
            #算還需要補足的部分佔下一個的portion是多少
            portion = round((30/5 - round_prev)/(speed_record_for_round[k-1]), 2)
            #print("portion:", portion)
            #(Index+portion-1)*5就是滿足30m的秒數
            round_time = round((k + portion - 1)*5, 1)
            round_record.append(round_time)
            
            round_over = 0
            #print("round over after minus:", round_over)
            speed_record_for_round[k-1] -= round((30/5 - round_prev))
            if speed_record_for_round[k-1] == 0:
                del speed_record_for_round[0:k]
            else:
                del speed_record_for_round[0:k-1]
            #前面剩下的速度乘5
            round_prev = 5*portion
            #print("speed_record_now:", speed_record_for_round)
            #print("round_record_now:", round_record)
            k = 0
        #每五秒計算一次，所以round_over還沒滿30/5=6時，就一直把記錄到的時間加進去
        else:
            round_over += speed_record_for_round[k]
            round_over = round(round_over, 2)
            k+=1
        
    time_list = []
    for j in range(len(round_record)):
        time_dict = {"Group":Group, "Date":Date, "ID":ID, "Round":j+1, "Time Spent(s)":round_record[j]}
        time_list.append(time_dict)

    for i in time_list:
        round_insert = Rounds(patient_id = i['ID'],
            group = i['Group'],
            date = i['Date'],
            time = i['Time Spent(s)'],
            round_count = i['Round']
            )
        db.session.add(round_insert)
    db.session.commit()

    round_group_query = db.session.query(Rounds.group.distinct().label("group_number"))
    group_list = [a.group_number for a in round_group_query.all()]
    df = pd.read_sql_query('select * from "rounds"',con=engine)
    for i in group_list:
        for j in range(1, 15):
            filter_func = Rounds.query.filter_by(patient_id = "average of group"+str(i), group = i, round_count= j).first()
            data = df[(df['group']==i) & (df['round_count']==j)]
            #print(data)
            time_data = round(data['time'].mean(), 2)
            #print("time data", time_data)
            if filter_func:
                filter_func.time = time_data
                db.session.commit()
            else:
                time_average = Rounds(patient_id = "average of group"+str(i), group=i, date = "2000-01-01", round_count=j, time=time_data)
                db.session.add(time_average)
                db.session.commit()
    return round_record

