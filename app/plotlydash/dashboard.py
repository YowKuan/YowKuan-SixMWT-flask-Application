from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import random
import numpy as np
import pandas as pd
from app.model.Distance_bar import Distance_bar
from app.model.Speed_chart import Speed_chart
from app.model.Rounds import Rounds
from app.model.Distance_speed import Distance_speed
from app.model.Video import Video
from sqlalchemy import create_engine

from os import environ, path
from dotenv import load_dotenv
basedir = path.abspath(path.dirname(__file__))
load_dotenv(path.join(basedir, '.env'))

SQLALCHEMY_DATABASE_URI = environ.get('SQLALCHEMY_DATABASE_URI')
engine = create_engine(SQLALCHEMY_DATABASE_URI)

def update_database():
    engine = create_engine(SQLALCHEMY_DATABASE_URI)
    df = pd.read_sql_query('select * from "speed_chart"',con=engine)
    df = df.sort_values(by=['id'])
    b = df['group'].unique()
    group_index = sorted(b)
    dff = pd.read_sql_query('select * from "rounds"',con=engine)
    dff = dff.sort_values(by=['id'])
    dfff = pd.read_sql_query('select * from "distance_bar"',con=engine)
    dfff = dfff.sort_values(by=['id'])
    dffff = pd.read_sql_query('select * from "distance_speed"',con=engine)
    dffff = dffff.sort_values(by=['id'])
    return df, dff, dfff, dffff, group_index

def init_dashboard(server):
    #Create a Plotly Dash dashboard
    df, dff, dfff, dffff, group_index = update_database()
    figg = px.line(dff, x = "round_count", y = "time", title = 'The 6-Minute Walking Test - Round Result')
    figg.update_traces(mode='markers+lines')

    patient_unique = df['patient_id'].unique()
    color_discrete_map = {}

    named_colorscales = ['blue', 'navy', 'red', 'maroon', 'purple',  
    'blueviolet', 'darkblue', 'forestgreen', 'grey', 'limegreen', 'mediumblue',
    'orchid', 'tan', 'tomato', 'slategrey', 'black', 'gold', 'lightskyblue', 'mediumspringgreen', 'orangered']
    
    dash_app = Dash(server = server, routes_pathname_prefix = '/dashapp/')
    dash_app.layout = html.Div("hello")
    dash_app.layout = html.Div([
        dcc.Graph(id='speed-plot'),
        dcc.Graph(id='speed-plot-1'),
        dcc.Dropdown(
            id ='groups',
            options=[{'label': s, 'value': s} for s in group_index],
            multi=True, 
            placeholder='Select the group'
        ),
        dcc.Interval(
            id='interval-component',
            interval=5*1000, # in milliseconds
            n_intervals=0
        ),
        dcc.Dropdown(id='patient-choice', options=[], multi=True),
        dcc.Graph(id='distance-plot'),
        dcc.Graph(id='distance-speed-plot'),
        dcc.Graph(id='dis-speed-mov-avg')
    ]) 
    
     
    @dash_app.callback(
        Output('patient-choice', 'options'),
        [Input('groups', 'value'), Input('interval-component', 'n_intervals')]
    )
    def filter_by_groups(groups, n):
        engine = create_engine(SQLALCHEMY_DATABASE_URI)
        dfff = pd.read_sql_query('select * from "distance_bar"',con=engine)
        for i in range(len(groups)):
            if i==0:
                filtered_group = dfff[(dfff['group']==groups[0])]
            filtered_group =  pd.concat([filtered_group,  dfff[(dfff['group']==groups[i])]])

        return [{'label': s, 'value': s} for s in filtered_group['patient_id'].unique()]



    @dash_app.callback(
    [Output('speed-plot', 'figure'), Output('distance-plot', 'figure'), Output('distance-speed-plot', 'figure'), Output('dis-speed-mov-avg', 'figure'), Output('speed-plot-1', 'figure')],
    [Input('groups', 'value'), Input('patient-choice', 'value'), Input('interval-component', 'n_intervals')]
    )
    def update_figure(selected_group, selected_id, n):
        df, dff, dfff, dffff, group_index = update_database()

        if len(selected_group)==1:
            speed_filtered_df = df[(df['group']==selected_group[0]) & (df['patient_id'].isin(selected_id))] 
        else:
            speed_filtered_df =df[df['patient_id'].isin(selected_id)]
        fig = px.line(speed_filtered_df, x = "time", y = "speed",color='patient_id', color_discrete_map=color_discrete_map, title = 'The 6-Minute Walking Test - Speed Result')
        fig.update_traces(mode='markers+lines')

        if len(selected_group)==1:
            dis_filtered_df = dfff[(dfff['group']==selected_group[0]) & (dfff['patient_id'].isin(selected_id))]
        else:
            dis_filtered_df = dfff[dfff['patient_id'].isin(selected_id)]
        figg = px.bar(dis_filtered_df, x="patient_id", y="distance", color='patient_id', color_discrete_sequence = px.colors.sequential.Plasma_r, title = 'Distance Comparison')

        if len(selected_group)==1:
            dis_time_df = dffff[(dffff['group']==selected_group[0]) & (dffff['patient_id'].isin(selected_id))]
        else:
            dis_time_df = dffff[dffff['patient_id'].isin(selected_id)]
        figgg = px.line(dis_time_df, x = "distance", y = "speed", color='patient_id',color_discrete_map=color_discrete_map, title = 'The 6-Minute Walking Test - Distance - Speed Chart')
        figgg = figgg.update_layout(
            xaxis = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 15
            )
        )
        figggg = px.line(dis_time_df, x = "distance", y='moving_avg', color='patient_id', color_discrete_map=color_discrete_map, title = 'The 6-Minute Walking Test - Distance - Speed Moving Average Chart')
        figggg = figggg.update_layout(
            xaxis = dict(
                tickmode = 'linear',
                tick0 = 0,
                dtick = 15
            ),
            yaxis = dict(
                tickmode = 'linear',
                tick0 = 0, 
                dtick = 0.5)
        )
        figggg = figggg.update_yaxes(range=[0.5, 3])
        dis_time_df = dis_time_df.sort_values(by=['time'])
        figu = px.line(dis_time_df, x ="time", y='moving_avg', color='patient_id', color_discrete_map=color_discrete_map, title = 'The 6-Minute Walking Test - time - Speed -1 Chart')
        return fig, figg, figgg, figggg, figu

    return dash_app.server