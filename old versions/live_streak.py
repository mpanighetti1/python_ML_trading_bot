######################## Imports #############################

# Oanda Imports
from turtle import width
import v20
from v20.order import MarketOrderRequest

# Data Imports
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from collections import namedtuple, deque
import os
import time
import json
import pytz

# Numba Imports
from numba import njit
from numba import prange

# Trading Imports
import talib
import vectorbtpro as vbt

# Dash Imports
import keyboard
import dash
from dash import dcc
from dash import html
from dash.dependencies import Input, Output
import plotly.graph_objects as go
import threading
import atexit
import dash_bootstrap_components as dbc

stop_event = threading.Event()
lock = threading.Lock()

##################### Timezone #######################

# Create a timezone object for EST
eastern = pytz.timezone('US/Eastern')

def adjust_fractional_seconds(timestamp):
    # Extract the fractional seconds part of the timestamp
    fractional_seconds = timestamp.split('.')[1].split('Z')[0]
    
    # Adjust the fractional seconds to always have a 6-digit precision
    while len(fractional_seconds) < 6:
        fractional_seconds += '0'
    fractional_seconds = fractional_seconds[:6]
    
    # Reconstruct the timestamp with adjusted fractional seconds
    adjusted_timestamp = timestamp.split('.')[0] + '.' + fractional_seconds + 'Z'
    return adjusted_timestamp

##################### Redis ##################################

import redis
    
redis_connection_1 = redis.Redis()
redis_connection_2 = redis.Redis()
redis_connection_3 = redis.Redis()
redis_connection_1.flushall()

def get_redis_connection(id):
    if id == 1:
        return redis_connection_1
    elif id == 2:
        return redis_connection_2
    elif id == 3:
        return redis_connection_3
    else:
        raise ValueError('Invalid connection ID')

def decode_data(r):
    data = r.get('data')
    if data is not None:
        data = json.loads(data)
        df = pd.DataFrame(data)
        if 'Open time' in df.columns:
            df['Open time'] = pd.to_datetime(df['Open time'], format="%Y-%m-%dT%H:%M:%S.%f%z")
            df['Open time'] = df['Open time'].dt.tz_convert('US/Eastern')
        return df
    return pd.DataFrame()

##################### Dash App #################################

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])

app.layout = html.Div([
    dcc.Interval(
        id='interval-component',
        interval=5*60*1000,  # in milliseconds
        n_intervals=0
    ),
    dcc.Graph(id='live-graph'),
])

@atexit.register
def stop_streaming():
    r = get_redis_connection(2)
    r.set('stop_streaming', 'True')
    r.flushall()
    redis_connection_1.close() # Close the Redis connections
    redis_connection_2.close()
    stop_event.set() # Set the stop event    
    
@app.callback(
    Output('live-graph', 'figure'),
    [Input('interval-component', 'n_intervals')]
)
def update_graph_live(n):
    
    # Create a new Redis connection
    r = get_redis_connection(1)

    with lock:        
        # Get Data From Redis
        df = decode_data(r)
        
    if df is None:
        return go.Figure()
    
    required_columns_initial = [
       'Open time', 'Open', 'High', 'Low', 'Close'
    ]
    
    # Create empty Figure
    fig = go.Figure()

    if all(col in df.columns for col in required_columns_initial):
        
        # Filter data for last 1000 minutes
        now = pd.Timestamp.now(tz='US/Eastern')
        cutoff = now - pd.Timedelta(minutes=1000)
        df = df.loc[df['Open time'] > cutoff]
        
        df_candles = df.drop_duplicates(subset=['Open time'], keep='last')

        fig.add_trace(go.Candlestick(
            x=df_candles['Open time'],
            open=df_candles['Open'],
            high=df_candles['High'],
            low=df_candles['Low'],
            close=df_candles['Close'],
            increasing_line_color='black',
            decreasing_line_color='black',
            increasing_fillcolor='white',
            decreasing_fillcolor='black',
            line=dict(width=0.5),
            name="Historical"
        ))

    # Column names you are looking for
    required_columns = [
        'short entry 1', 'short entry 2', 'short entry 3', 
        'long entry 1', 'long entry 2', 'long entry 3', 
        'short stop-loss hit', 'long stop-loss hit', 
        'short take-profit hit', 'long take-profit hit', 
        'bids', 'asks', 'short cond', 'long cond',
        'short entry 1 level', 'short entry 2 level', 
        'short entry 3 level', 'long entry 1 level', 
        'long entry 2 level', 'long entry 3 level', 
        's1', 's2', 's3', 'short stop-loss', 'short take-profit', 
        'l1', 'l2', 'l3', 'long stop-loss', 'long take-profit'
    ]
    
    if all(col in df.columns for col in required_columns):
        
        # Define level_colors
        long_colors = ['limegreen', 'green', 'darkgreen']
        short_colors = ['red', 'firebrick', 'darkred']
        
        for i in range(1, 4):
            if df[f'short entry {i}'].fillna(False).any():
                fig.add_trace(go.Scatter(
                    x=df.loc[df[f'short entry {i}'].fillna(False), 'Open time'], 
                    y=df.loc[df[f'short entry {i}'].fillna(False), 'bids'], 
                    mode='markers',
                    marker=dict(color=short_colors[i-1], symbol='triangle-down', size=10), # down-triangles for short
                    name=f'Short entry {i}'
                ))
            if df[f'long entry {i}'].fillna(False).any():
                fig.add_trace(go.Scatter(
                    x=df.loc[df[f'long entry {i}'].fillna(False), 'Open time'], 
                    y=df.loc[df[f'long entry {i}'].fillna(False), 'asks'], 
                    mode='markers',
                    marker=dict(color=long_colors[i-1], symbol='triangle-up', size=10), # up-triangles for long
                    name=f'Long entry {i}'
                ))

        if df['short stop-loss hit'].fillna(False).any():
            fig.add_trace(go.Scatter(
                x=df.loc[df['short stop-loss hit'].fillna(False), 'Open time'], 
                y=df.loc[df['short stop-loss hit'].fillna(False), 'asks'], 
                mode='markers',
                marker=dict(color='darkviolet', size=10), 
                name='Short stop-loss hit'
            ))

        if df['long stop-loss hit'].fillna(False).any():
            fig.add_trace(go.Scatter(
                x=df.loc[df['long stop-loss hit'].fillna(False), 'Open time'], 
                y=df.loc[df['long stop-loss hit'].fillna(False), 'bids'], 
                mode='markers',
                marker=dict(color='darkviolet', size=10), 
                name='Long stop-loss hit'
            ))

        if df['short take-profit hit'].fillna(False).any():
            fig.add_trace(go.Scatter(
                x=df.loc[df['short take-profit hit'].fillna(False), 'Open time'], 
                y=df.loc[df['short take-profit hit'].fillna(False), 'asks'], 
                mode='markers',
                marker=dict(color='dodgerblue', size=10), 
                name='Short take-profit hit'
            ))

        if df['long take-profit hit'].fillna(False).any():
            fig.add_trace(go.Scatter(
                x=df.loc[df['long take-profit hit'].fillna(False), 'Open time'], 
                y=df.loc[df['long take-profit hit'].fillna(False), 'bids'], 
                mode='markers',
                marker=dict(color='dodgerblue', size=10), 
                name='Long take-profit hit'
            ))

        for i in range(1, 4):
            y_values = df[f'short entry {i} level'].where(df['short cond']).tolist()
            x_values = df['Open time'].tolist()
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines',
                line_shape='hv',
                line=dict(color=short_colors[i-1]),
                name=f'Short entry {i} level'
            ))
            
        for i in range(1, 4):
            y_values = df[f'long entry {i} level'].where(df['long cond']).tolist()
            x_values = df['Open time'].tolist()
            fig.add_trace(go.Scatter(
                x=x_values,
                y=y_values,
                mode='lines',
                line_shape='hv',
                line=dict(color=long_colors[i-1]),
                name=f'Long entry {i} level'
            ))
                
        short_condition = df['s1'] | df['s2'] | df['s3']
        fig.add_trace(go.Scatter(
            x=df.loc[short_condition, 'Open time'],
            y=df.loc[short_condition, 'short stop-loss'],
            mode='lines',
            line_shape='hv',
            line=dict(color='darkviolet'),
            name='Short stop-loss'
        ))
        fig.add_trace(go.Scatter(
            x=df.loc[short_condition, 'Open time'],
            y=df.loc[short_condition, 'short take-profit'],
            mode='lines',
            line_shape='hv',
            line=dict(color='dodgerblue'),
            name='Short take-profit'
        ))

        long_condition = df['l1'] | df['l2'] | df['l3']
        fig.add_trace(go.Scatter(
            x=df.loc[long_condition, 'Open time'],
            y=df.loc[long_condition, 'long stop-loss'],
            mode='lines',
            line_shape='hv',
            line=dict(color='darkviolet'),
            name='Long stop-loss'
        ))
        fig.add_trace(go.Scatter(
            x=df.loc[long_condition, 'Open time'],
            y=df.loc[long_condition, 'long take-profit'],
            mode='lines',
            line_shape='hv',
            line=dict(color='dodgerblue'),
            name='Long take-profit'
        ))

    # Laptop Dimensions (Chrome) Width: 1365 Height: 923
    # Desktop Dimensions (Chrome) Height: 1291
    fig.update_layout(
        plot_bgcolor='lightgrey',
        paper_bgcolor='lightgrey',
        font=dict(color='black'),
        hovermode='x',
        autosize=True,  # Set autosize to False
        #width=1365,      # Set the fixed width
        height=1291,      # Set the fixed height
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis=dict(
            zeroline=False,
            showticklabels=True,
            showgrid=False,
            rangeslider=dict(visible=False) 
        ),
        yaxis=dict(
            showgrid=False
        )
    )
    
    return fig

##################### API Authentication ########################

# Authenticate to OANDA
access_token = '650d35fdf3fdb30d1056bcfe6aa9ae24-7ba8cbb7e3759f01e7b8b2775f7d9b8d'
account_id = '001-001-8347571-001'
hostname = 'api-fxtrade.oanda.com'
streaming_hostname = 'stream-fxtrade.oanda.com'
instrument = 'EUR_USD'

# Create an API context
api = v20.Context(hostname, 443, token=access_token)
streaming_api = v20.Context(streaming_hostname, 443, token=access_token)

##################### Fetch Candles #############################

def fetch_candles(instrument, granularity, count):
    delay = 5     # Seconds to wait between retries

    while True:
        try:
            params = {
                "granularity": granularity,
                "count": count,
            }

            # Fetch the candles
            response = api.instrument.candles(instrument, **params)

            if response.status != 200:
                print(f"Error: {response.body}")
                return None

            candles = response.body.get("candles", [])
            if not candles:
                print("No candle data found")
                return None

            df = pd.DataFrame([{
                "Open time": pd.to_datetime(candle.time),
                "Open": float(candle.mid.o),
                "High": float(candle.mid.h),
                "Low": float(candle.mid.l),
                "Close": float(candle.mid.c),
                "Volume": int(candle.volume),
                "Complete": bool(candle.complete)
            } for candle in candles[:-1]])  # Exclude the last candle

            df['Open time'] = df['Open time'].dt.tz_convert('US/Eastern')
            df = df.set_index('Open time')  # Set the 'Open time' column as the index

            # Convert granularity to pandas frequency
            freq_map = {'M1': '1min', 'M15': '15min'}
            freq = freq_map.get(granularity, granularity)
            
            # Resample to the granularity and forward-fill any gaps
            df = df.resample(freq).ffill()

            # Fill any remaining NaN values (optional)
            df = df.fillna(method='ffill')

            # Convert columns to float64 if needed (optional)
            df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']].astype(np.float64)

            return df

        except Exception as e:
            print(f"Error occurred: {e}. Retrying in {delay} seconds...")
            time.sleep(delay)       # Wait for 5 seconds before retrying

##################### Ensure 2-Dim Arrays #######################

def ensure_2d(arr):
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

#################### Trading Parameters #########################

'''
# Last Updated 9/16/2023
Top trial information from the last study:

Objective values:
Objective 1: 0.036807
Objective 2: 200.0

Constraint values:
Constraint 1: -0.009504151403093113
Constraint 2: -303.33119807827046
Constraint 3: 0.0
Constraint 4: -0.03680701362313891
'''
first_target_length_input = 542
rsi_lower_threshold = 60
rsi_period = 212
rsi_upper_threshold = 53
sl_multiplier = 6.0
sl_period = 247
short_target_multiple = 0.00029
long_target_multiple  = 0.00252
short_entry_multiple  = 0.00008
long_entry_multiple   = 0.00001
will_frac_period = 2

##################### Indicator Functions #######################

##################### ATR (Stop Loss) ###########################
def get_sl_atr(high, low, close, period, multiplier):
    sl_atr = vbt.talib('ATR').run(high, low, close, timeperiod=period, skipna=True).real.to_numpy().astype(np.float64)
    short_sl_atr_value, long_sl_atr_value = get_sl_atr_nb(sl_atr, close, multiplier)
    return short_sl_atr_value, long_sl_atr_value

@njit(nogil=True)
def get_sl_atr_nb(sl_atr, close, multiplier):
    sl_scaled_atr = sl_atr * multiplier
    short_sl_atr_value = close + sl_scaled_atr
    long_sl_atr_value = close - sl_scaled_atr
    return short_sl_atr_value, long_sl_atr_value

##################### RSI Indicator #############################
def get_rsi(close, period, upper_threshold, lower_threshold):
    rsi = vbt.talib('RSI').run(close, timeperiod=period, skipna=True).real.to_numpy().astype(np.float64)
    above_upper_threshold, below_lower_threshold = get_rsi_nb(rsi, upper_threshold, lower_threshold)
    return above_upper_threshold, below_lower_threshold

@njit(nogil=True)
def get_rsi_nb(rsi, upper_threshold, lower_threshold):
    above_upper_threshold = rsi >= upper_threshold
    below_lower_threshold = rsi <= lower_threshold
    return above_upper_threshold, below_lower_threshold

######### Williams Fractals Helper Functions ###################

def find_highest_lowest(high, low, first_target_length_input):
    highest_of_past_length = vbt.talib('MAX').run(high, timeperiod=first_target_length_input, skipna=True).real.to_numpy().astype(np.float64)
    lowest_of_past_length = vbt.talib('MIN').run(low, timeperiod=first_target_length_input, skipna=True).real.to_numpy().astype(np.float64)
    return highest_of_past_length, lowest_of_past_length

@njit(nogil=True)
def will_frac(high, 
              low, 
              highest_of_past_length, 
              lowest_of_past_length, 
              fractal_period, 
              long_target_multiple, 
              short_target_multiple, 
              long_entry_multiple, 
              short_entry_multiple):
    
    botFractals = np.full(low.shape, False)
    topFractals = np.full(high.shape, False)

    long_take_profit_value = np.full(high.shape, np.nan)
    short_take_profit_value = np.full(low.shape, np.nan)

    short_entry_max_value = np.full(high.shape, np.nan)
    short_entry_min_value = np.full(high.shape, np.nan)
    short_entry_mid_value = np.full(high.shape, np.nan)    

    long_entry_max_value = np.full(low.shape, np.nan)
    long_entry_min_value = np.full(low.shape, np.nan)
    long_entry_mid_value = np.full(low.shape, np.nan)

    short_entry_first = np.nan
    short_entry_second = np.nan
    short_entry_third = np.nan

    long_entry_first = np.nan
    long_entry_second = np.nan
    long_entry_third = np.nan

    for n in range(1, high.shape[0]): # range(1, len(h)):

        upflagDownFrontier = True
        upflagUpFrontier0 = True
        upflagUpFrontier1 = True
        upflagUpFrontier2 = True
        upflagUpFrontier3 = True
        upflagUpFrontier4 = True

        downflagDownFrontier = True
        downflagUpFrontier0 = True
        downflagUpFrontier1 = True
        downflagUpFrontier2 = True
        downflagUpFrontier3 = True
        downflagUpFrontier4 = True

        for i in range(1, fractal_period + 1):
            upflagDownFrontier = upflagDownFrontier and (high[n-(fractal_period-i)] < high[n-fractal_period])
            upflagUpFrontier0 = upflagUpFrontier0 and (high[n-(fractal_period+i)] < high[n-fractal_period])
            upflagUpFrontier1 = upflagUpFrontier1 and (high[n-(fractal_period+1)] <= high[n-fractal_period] and 
                                                       high[n-(fractal_period+i+1)] < high[n-fractal_period])                                                       
            upflagUpFrontier2 = upflagUpFrontier2 and (high[n-(fractal_period+1)] <= high[n-fractal_period] and 
                                                       high[n-(fractal_period+2)] <= high[n-fractal_period] and 
                                                       high[n-(fractal_period+i+2)] < high[n-fractal_period])                                                   
            upflagUpFrontier3 = upflagUpFrontier3 and (high[n-(fractal_period+1)] <= high[n-fractal_period] and
                                                       high[n-(fractal_period+2)] <= high[n-fractal_period] and
                                                       high[n-(fractal_period+3)] <= high[n-fractal_period] and
                                                       high[n-(fractal_period+i+3)] < high[n-fractal_period])
            upflagUpFrontier4 = upflagUpFrontier4 and (high[n-(fractal_period+1)] <= high[n-fractal_period] and
                                                       high[n-(fractal_period+2)] <= high[n-fractal_period] and 
                                                       high[n-(fractal_period+3)] <= high[n-fractal_period] and
                                                       high[n-(fractal_period+4)] <= high[n-fractal_period] and
                                                       high[n-(fractal_period+i+4)] < high[n-fractal_period])

            downflagDownFrontier = downflagDownFrontier and (low[n-(fractal_period-i)] > low[n-fractal_period])
            downflagUpFrontier0 = downflagUpFrontier0 and (low[n-(fractal_period+i)] > low[n-fractal_period])
            downflagUpFrontier1 = downflagUpFrontier1 and (low[n-(fractal_period+1)] >= low[n-fractal_period] and
                                                           low[n-(fractal_period+i+1)] > low[n-fractal_period])          
            downflagUpFrontier2 = downflagUpFrontier2 and (low[n-(fractal_period+1)] >= low[n-fractal_period] and
                                                           low[n-(fractal_period+2)] >= low[n-fractal_period] and
                                                           low[n-(fractal_period+i+2)] > low[n-fractal_period])
            downflagUpFrontier3 = downflagUpFrontier3 and (low[n-(fractal_period+1)] >= low[n-fractal_period] and
                                                           low[n-(fractal_period+2)] >= low[n-fractal_period] and 
                                                           low[n-(fractal_period+3)] >= low[n-fractal_period] and 
                                                           low[n-(fractal_period+i+3)] > low[n-fractal_period])
            downflagUpFrontier4 = downflagUpFrontier4 and (low[n-(fractal_period+1)] >= low[n-fractal_period] and 
                                                           low[n-(fractal_period+2)] >= low[n-fractal_period] and 
                                                           low[n-(fractal_period+3)] >= low[n-fractal_period] and 
                                                           low[n-(fractal_period+4)] >= low[n-fractal_period] and 
                                                           low[n-(fractal_period+i+4)] > low[n-fractal_period])

        flagUpFrontier = (upflagUpFrontier0 or upflagUpFrontier1 or upflagUpFrontier2 or upflagUpFrontier3 or upflagUpFrontier4)

        if (upflagDownFrontier and flagUpFrontier):
            botFractals[n-fractal_period] = True
            short_entry_third = short_entry_second
            short_entry_second = short_entry_first
            short_entry_first = high[n-fractal_period]
            long_take_profit_value[n] = highest_of_past_length[n] * (1 + long_target_multiple)

            if (not np.isnan(short_entry_first) and not np.isnan(short_entry_second) and not np.isnan(short_entry_third)):
                short_entry_max_value[n] = (max(short_entry_first, short_entry_second, short_entry_third)) * (1 + short_entry_multiple)
                short_entry_min_value[n] = (min(short_entry_first, short_entry_second, short_entry_third)) * (1 + short_entry_multiple)
                short_entry_mid_value[n] = ((short_entry_first + short_entry_second + short_entry_third - short_entry_max_value[n] - short_entry_min_value[n])) * (1 + short_entry_multiple)
        
        if (not botFractals[n-fractal_period]): # if botFractals[n] == False, not False = True  
            short_entry_max_value[n] = short_entry_max_value[n-1]
            short_entry_min_value[n] = short_entry_min_value[n-1]
            short_entry_mid_value[n] = short_entry_mid_value[n-1]
            long_take_profit_value[n] = long_take_profit_value[n-1]        
            
        flagDownFrontier = (downflagUpFrontier0 or downflagUpFrontier1 or downflagUpFrontier2 or downflagUpFrontier3 or downflagUpFrontier4)

        if (downflagDownFrontier and flagDownFrontier):
            topFractals[n-fractal_period] = True
            long_entry_third = long_entry_second
            long_entry_second = long_entry_first
            long_entry_first = low[n-fractal_period]
            short_take_profit_value[n] = lowest_of_past_length[n] * (1 - short_target_multiple)

            if (not np.isnan(long_entry_first) and not np.isnan(long_entry_second) and not np.isnan(long_entry_third)):
                long_entry_max_value[n] = (max(long_entry_first, long_entry_second, long_entry_third)) * (1 - long_entry_multiple)
                long_entry_min_value[n] = (min(long_entry_first, long_entry_second, long_entry_third)) * (1 - long_entry_multiple)
                long_entry_mid_value[n] = (long_entry_first + long_entry_second + long_entry_third - long_entry_max_value[n] - long_entry_min_value[n]) * (1 - long_entry_multiple)

        if (not topFractals[n-fractal_period]): # if topFractals[n] == False, not False = True         
            long_entry_max_value[n] = long_entry_max_value[n-1]
            long_entry_min_value[n] = long_entry_min_value[n-1]
            long_entry_mid_value[n] = long_entry_mid_value[n-1]
            short_take_profit_value[n] = short_take_profit_value[n-1]           

    return (botFractals, 
            topFractals, 
            long_take_profit_value, 
            short_entry_max_value, 
            short_entry_min_value, 
            short_entry_mid_value, 
            short_take_profit_value, 
            long_entry_max_value, 
            long_entry_min_value, 
            long_entry_mid_value)

######################### Liquidity Purge Strategy ############################
  
expr = """
LiquidityPurgeStrategy[lps]:

# ATR Stop Loss
(short_sl_atr_value,
long_sl_atr_value) = get_sl_atr(@in_LTF_high, 
                                @in_LTF_low, 
                                @in_LTF_close, 
                                @p_sl_period, 
                                @p_sl_multiplier)

# Williams Fractals
highest_of_past_length, lowest_of_past_length = find_highest_lowest(@in_LTF_high, @in_LTF_low, @p_first_target_length_input)

(topFractals, 
botFractals, 
long_take_profit_value, 
short_entry_max_value, 
short_entry_min_value, 
short_entry_mid_value, 
short_take_profit_value, 
long_entry_max_value, 
long_entry_min_value, 
long_entry_mid_value) = will_frac(@in_LTF_high, 
                        @in_LTF_low, 
                        highest_of_past_length, 
                        lowest_of_past_length, 
                        @p_will_frac_period, 
                        @p_short_target_multiple,
                        @p_long_target_multiple,
                        @p_short_entry_multiple,
                        @p_long_entry_multiple)

# Returns
long_sl_atr_value, short_sl_atr_value, \
topFractals, botFractals, \
long_take_profit_value, short_entry_max_value, short_entry_min_value, short_entry_mid_value, \
short_take_profit_value, long_entry_max_value, long_entry_min_value, long_entry_mid_value

"""

LiquidityPurgeStrategy = vbt.IF.from_expr(
    expr,
    takes_1d=True,

    # ATR Stop Loss
    sl_period=sl_period,
    sl_multiplier=sl_multiplier,
    get_sl_atr=get_sl_atr,

    # Williams Fractals
    first_target_length_input=first_target_length_input,
    will_frac_period=will_frac_period,
    short_target_multiple=short_target_multiple,
    long_target_multiple=long_target_multiple,
    short_entry_multiple=short_entry_multiple,
    long_entry_multiple=long_entry_multiple,
    find_highest_lowest=find_highest_lowest,
    will_frac=will_frac    
)

######################### RSI Indicator ##################################

expr = """
RelativeStrengthIndex[rsi]:

# RSI Indicator
rsi_above_upper_threshold, rsi_below_lower_threshold = get_rsi(@in_LTF_close, 
                                                                @p_rsi_period,
                                                                @p_rsi_upper_threshold, 
                                                                @p_rsi_lower_threshold)

# Returns
rsi_above_upper_threshold, rsi_below_lower_threshold

"""

RelativeStrengthIndex = vbt.IF.from_expr(
    expr,
    takes_1d=True,
    
    # RSI Indicator
    rsi_period=rsi_period,
    rsi_upper_threshold=rsi_upper_threshold,
    rsi_lower_threshold=rsi_lower_threshold,
    get_rsi=get_rsi, 
)

#################### Execution Functions ############################

def get_account_balance(max_attempts=5):
    attempts = 0
    while attempts < max_attempts:
        try:
            account_summary = api.account.get(account_id)
            return float(account_summary.get('account').balance) * 50 # Leverage
        except Exception as e:
            attempts += 1
            time.sleep(1) # Wait for 1 seconds before retrying. You can adjust this as needed.
    raise Exception(f"Failed to get account balance after {max_attempts} attempts.")

def get_instrument_price(instrument):
    try:
        pricing_info = api.pricing.get(account_id, instruments=instrument)
        prices = pricing_info.get('prices')
        return (float(prices[0].bids[0].price) + float(prices[0].asks[0].price)) / 2.0
    except Exception as e:
        return 0.0

def calculate_units(instrument, percent, is_long):
    try:
        # Get account balance
        balance = float(get_account_balance())
        # Calculate amount to invest
        amount = balance * percent
        # Get current instrument price
        price = float(get_instrument_price(instrument))
        # Calculate units
        units = int(amount / price)
        # If the position is short, make units negative
        if not is_long:
            units = -units
        return units
    except Exception as e:
        return 0

def execute_order(instrument, percent, is_long):
        # Calculate units based on account balance and percent
        units = calculate_units(instrument, percent, is_long)
        direction = 'LONG' if is_long else 'SHORT'
        # Execute the order
        response = api.order.market(account_id, instrument=instrument, units=units)

def close_all_positions():
        # List all open trades
        trades = api.trade.list_open(account_id).body.get("trades", [])
        # Close all open trades
        for trade in trades:
            api.trade.close(account_id, trade.id)

#################### Streaming Task #####################################

# Shared data structure
latest_msgs = deque(maxlen=1)  # deque with maxlen=1 will only store the latest value

def streaming_task():
    
    r = get_redis_connection(2)
    
    while r.get('stop_streaming') != 'True':
        if stop_event.is_set(): 
            break
        try:        
            try:
                stream = streaming_api.pricing.stream(account_id, snapshot=True, instruments=instrument)
                print("Streaming service set up correctly.")
            except Exception as e:
                print(f"Error setting up streaming service: {str(e)}")

            try:
                test_msg_type, test_msg = next(iter(stream.parts()))
                print(f"Test message type: {test_msg_type}")
                print(f"Test message: {test_msg}")
            except Exception as e:
                print(f"Error retrieving data from stream: {str(e)}")
                
            print("Starting streaming and trading loop...")
    
            while True:
                for msg_type, msg in stream.parts():
                    if msg_type == "pricing.PricingHeartbeat":
                        pass
                    elif msg_type == "pricing.ClientPrice":
                        latest_msgs.append(msg)  # deque will only keep the latest message
                        
        except Exception as e:
            print(f"Error occurred: {str(e)}. Reconnecting...")

        # Add some sleep to avoid rapid-fire reconnection attempts
        time.sleep(5)

def main_loop():
    
    r = get_redis_connection(3)
    
    #################### Starting Values ####################################

    LTF_granularity = 'M1' # This is the period of everything else
    mid_price = np.nan # This will store the latest mid price
    last_mid_price = np.nan # This will store the previous mid price.
    current_long_stop_loss_value = np.nan  # This will store the latest long stop loss value
    current_short_stop_loss_value = np.nan  # This will store the latest short stop loss value
    current_long_take_profit_value = np.nan  # This will store the latest long take profit value
    current_short_take_profit_value = np.nan  # This will store the latest short take profit value
    LTF = fetch_candles(instrument, LTF_granularity, 3000) # highest period of everything else
    last_minute = datetime.now().minute
    
    LTF_open = LTF['Open']
    LTF_high = LTF['High']
    LTF_low = LTF['Low']
    LTF_close = LTF['Close']

    lps = LiquidityPurgeStrategy.run(LTF_high, LTF_low, LTF_close,
        sl_period=sl_period,
        sl_multiplier=sl_multiplier,
        first_target_length_input=first_target_length_input,
        will_frac_period=will_frac_period,
        short_target_multiple=short_target_multiple,
        long_target_multiple=long_target_multiple,
        short_entry_multiple=short_entry_multiple,
        long_entry_multiple=long_entry_multiple,
        execute_kwargs=dict(
            engine='dask',
            chunk_len='auto',
            show_progress=False
            ))

    rsi = RelativeStrengthIndex.run(LTF_close, # calls LTF_close
        rsi_period=rsi_period,
        rsi_upper_threshold=rsi_upper_threshold,
        rsi_lower_threshold=rsi_lower_threshold,                   
        execute_kwargs=dict(                                        
                        engine='dask',
                        chunk_len='auto',
                        show_progress=False
                        ))
    
    #################### Get Latest Indicator Values #################
    
    rsi_above_upper_threshold = rsi.rsi_above_upper_threshold.iloc[-1].item()
    rsi_below_lower_threshold = rsi.rsi_below_lower_threshold.iloc[-1].item()
    
    lps_short_entry_min_value = lps.short_entry_min_value.iloc[-1].item()
    lps_short_entry_mid_value = lps.short_entry_mid_value.iloc[-1].item()
    lps_short_entry_max_value = lps.short_entry_max_value.iloc[-1].item()

    lps_long_entry_min_value = lps.long_entry_min_value.iloc[-1].item()
    lps_long_entry_mid_value = lps.long_entry_mid_value.iloc[-1].item()
    lps_long_entry_max_value = lps.long_entry_max_value.iloc[-1].item()

    lps_short_take_profit_value = lps.short_take_profit_value.iloc[-1].item()
    lps_long_take_profit_value = lps.long_take_profit_value.iloc[-1].item()

    lps_short_sl_atr_value = lps.short_sl_atr_value.iloc[-1].item()
    lps_long_sl_atr_value = lps.long_sl_atr_value.iloc[-1].item()

    # Initialize a Series with two elements both set to 0
    mid_prices = pd.Series([0, 0])

    # Starting Values
    s1 = False
    s2 = False
    s3 = False  
    l1 = False
    l2 = False
    l3 = False
    short_stop_1 = 0 # It won't ever be these values - overwritten and sorted up entry.
    short_stop_2 = 0
    short_stop_3 = 0
    long_stop_1 = 1e20 
    long_stop_2 = 1e20
    long_stop_3 = 1e20
    short_tp_1 = 1e20
    short_tp_2 = 1e20
    short_tp_3 = 1e20
    long_tp_1 = 0
    long_tp_2 = 0
    long_tp_3 = 0
    long_tp_hit = False
    long_sl_hit = False
    short_tp_hit = False
    short_sl_hit = False
    long_entry_1_hit = False
    long_entry_2_hit = False
    long_entry_3_hit = False
    short_entry_1_hit = False
    short_entry_2_hit = False
    short_entry_3_hit = False
    long_cond = False
    short_cond = True

    with lock:
        # Update Redis data
        data = decode_data(r)

    for i in range(len(LTF)):
        df = pd.DataFrame({
            'Open time': [LTF.index[i]], # x-axis
            'Open': [LTF['Open'].iloc[i]], # 1-minute candle open
            'High': [LTF['High'].iloc[i]], # 1-minute candle high
            'Low': [LTF['Low'].iloc[i]], # 1-minute candle low
            'Close': [LTF['Close'].iloc[i]], # 1-minute candle close
            # add other data fields here as necessary
        })
        data = pd.concat([data, df], ignore_index=True)
        
    # Convert all Timestamp objects to strings
    data = data.applymap(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f%z") if isinstance(x, pd.Timestamp) else x)

    with lock:
        # Save data back into Redis
        r.set('data', json.dumps(data.to_dict('records')))

    ########################################### Stream Processing Loop ###########################################

    while r.get('stop_streaming') != 'True':                
        if latest_msgs:
            latest_msg = latest_msgs.popleft() # Process latest_msg
                
            ######################## Main Loop Starts #######################

            mid_price = (latest_msg.bids[0].price + latest_msg.asks[0].price) / 2

            mid_prices[0] = last_mid_price
            mid_prices[1] = mid_price
            last_mid_price = mid_price
            
            #################### Get Data For Indicators ####################
            
            # Trim the string to limit the precision to 6 digits
            adjusted_time = adjust_fractional_seconds(latest_msg.time)

            # Convert the adjusted string to a datetime object
            utc_time = datetime.strptime(adjusted_time, '%Y-%m-%dT%H:%M:%S.%fZ')
            utc_time = utc_time.replace(tzinfo=pytz.utc)  # Set its timezone to UTC
            
            # Convert UTC to EST
            est_time = utc_time.astimezone(eastern)

            # Fetch 1-minute data at the start of each new minute
            current_minute = datetime.now().minute
            if current_minute != last_minute:
                print(f"{est_time.strftime('%Y-%m-%d %I:%M:%S %p')}, Fetching New LTF Data...")
                LTF = fetch_candles(instrument, LTF_granularity, 3000) # highest period of everything else
                LTF_open = LTF['Open']
                LTF_high = LTF['High']
                LTF_low = LTF['Low']
                LTF_close = LTF['Close']
                
                lps = LiquidityPurgeStrategy.run(LTF_high, LTF_low, LTF_close,
                        sl_period=sl_period,
                        sl_multiplier=sl_multiplier,
                        first_target_length_input=first_target_length_input,
                        will_frac_period=will_frac_period,
                        short_target_multiple=short_target_multiple,
                        long_target_multiple=long_target_multiple,
                        short_entry_multiple=short_entry_multiple,
                        long_entry_multiple=long_entry_multiple,
                        execute_kwargs=dict(
                            engine='dask',
                            chunk_len='auto',
                            show_progress=False
                            ))
            
                rsi = RelativeStrengthIndex.run(LTF_close, # calls LTF_close
                    rsi_period=rsi_period,
                    rsi_upper_threshold=rsi_upper_threshold,
                    rsi_lower_threshold=rsi_lower_threshold,                   
                    execute_kwargs=dict(                                        
                                    engine='dask',
                                    chunk_len='auto',
                                    show_progress=False
                                    ))
                
            last_minute = current_minute
            
            rsi_above_upper_threshold = rsi.rsi_above_upper_threshold.iloc[-1].item()
            rsi_below_lower_threshold = rsi.rsi_below_lower_threshold.iloc[-1].item()
            
            lps_short_entry_min_value = lps.short_entry_min_value.iloc[-1].item()
            lps_short_entry_mid_value = lps.short_entry_mid_value.iloc[-1].item()
            lps_short_entry_max_value = lps.short_entry_max_value.iloc[-1].item()

            lps_long_entry_min_value = lps.long_entry_min_value.iloc[-1].item()
            lps_long_entry_mid_value = lps.long_entry_mid_value.iloc[-1].item()
            lps_long_entry_max_value = lps.long_entry_max_value.iloc[-1].item()

            lps_short_take_profit_value = lps.short_take_profit_value.iloc[-1].item()
            lps_long_take_profit_value = lps.long_take_profit_value.iloc[-1].item()

            lps_short_sl_atr_value = lps.short_sl_atr_value.iloc[-1].item()
            lps_long_sl_atr_value = lps.long_sl_atr_value.iloc[-1].item()      

            ################### Entry/Exit Signals ##########################

            # Short Entries      
            short_rsi_cond = rsi_above_upper_threshold
            
            short_entry_1_default_cond = (mid_prices.vbt.crossed_above(lps_short_entry_min_value).iloc[-1].item()) & short_rsi_cond
            short_entry_2_default_cond = (mid_prices.vbt.crossed_above(lps_short_entry_mid_value).iloc[-1].item()) & short_rsi_cond
            short_entry_3_default_cond = (mid_prices.vbt.crossed_above(lps_short_entry_max_value).iloc[-1].item()) & short_rsi_cond

            # Long Entries
            long_rsi_cond = rsi_below_lower_threshold

            long_entry_1_default_cond = (mid_prices.vbt.crossed_below(lps_long_entry_max_value).iloc[-1].item()) & long_rsi_cond
            long_entry_2_default_cond = (mid_prices.vbt.crossed_below(lps_long_entry_mid_value).iloc[-1].item()) & long_rsi_cond
            long_entry_3_default_cond = (mid_prices.vbt.crossed_below(lps_long_entry_min_value).iloc[-1].item()) & long_rsi_cond

            ################### Executing Trading Logic ####################

            # Check if the spread is too high
            spread = latest_msg.asks[0].price - latest_msg.bids[0].price
            if spread > 0.0002:
                print(f"{latest_msg.instrument} price update: {est_time.strftime('%Y-%m-%d %I:%M:%S %p')}, mid: {mid_price} - Spread Too High: ({spread})")
                continue
            
            if (short_cond):
                msg_dir = "Looking For Short Entry"
            elif (long_cond):
                msg_dir = "Looking For Long Entry"            
            
            if (l1 or l2 or l3):
                print(f"{latest_msg.instrument} price update: {est_time.strftime('%Y-%m-%d %I:%M:%S %p')}, mid: {mid_price} - Long Position Active")
            elif (s1 or s2 or s3):
                print(f"{latest_msg.instrument} price update: {est_time.strftime('%Y-%m-%d %I:%M:%S %p')}, mid: {mid_price} - Short Position Active")
            else:
                print(f"{latest_msg.instrument} price update: {est_time.strftime('%Y-%m-%d %I:%M:%S %p')}, mid: {mid_price} - No Position - {msg_dir}")
            
            ######################## LONGS ############################

            # Long Entries
            if long_entry_1_default_cond and \
            long_cond and \
            (not l1) and \
            (not (s1 or s2 or s3)) and \
            (not np.isnan(lps_long_sl_atr_value)) and \
            (not np.isnan(lps_long_take_profit_value)):               
                l1 = True # Set Long Entry 1 Position to True                
                # Long Entry 1 Trade Happens Here
                print(f"Long Entry 1 Executed: {latest_msg.instrument} Price: {mid_price}")
                execute_order(instrument, 0.33, is_long=True)                
                long_stop_1 = lps_long_sl_atr_value # Set Stop Loss 1 to ATR Stop Loss Value                
                # Set Stop Loss Value to the min of the 3 stop loss values
                current_long_stop_loss_value = min(long_stop_1, min(long_stop_2, long_stop_3))
                long_tp_1 = lps_long_take_profit_value       
                current_long_take_profit_value = max(long_tp_1, max(long_tp_2, long_tp_3))
                long_entry_1_hit = True
                
            if long_entry_2_default_cond and \
            long_cond and \
            (not l2) and \
            (not (s1 or s2 or s3)) and \
            (not np.isnan(lps_long_sl_atr_value)) and \
            (not np.isnan(lps_long_take_profit_value)):   
                l2 = True # Set Long Entry 2 Position to True
                # Long Entry 2 Trade Happens Here
                print(f"Long Entry 2 Executed: {latest_msg.instrument} Price: {mid_price}")
                execute_order(instrument, 0.33, is_long=True)                
                long_stop_2 = lps_long_sl_atr_value    
                current_long_stop_loss_value = min(long_stop_1, min(long_stop_2, long_stop_3))
                long_tp_2 = lps_long_take_profit_value       
                current_long_take_profit_value = max(long_tp_1, max(long_tp_2, long_tp_3))
                long_entry_2_hit = True
                
            if long_entry_3_default_cond and \
            long_cond and \
            (not l3) and \
            (not (s1 or s2 or s3)) and \
            (not np.isnan(lps_long_sl_atr_value)) and \
            (not np.isnan(lps_long_take_profit_value)):                
                l3 = True # Set Long Entry 3 Position to True
                # Long Entry 3 Trade Happens Here
                print(f"Long Entry 3 Executed: {latest_msg.instrument} Price: {mid_price}")
                execute_order(instrument, 0.33, is_long=True)                
                long_stop_3 = lps_long_sl_atr_value           
                current_long_stop_loss_value = min(long_stop_1, min(long_stop_2, long_stop_3))
                long_tp_3 = lps_long_take_profit_value       
                current_long_take_profit_value = max(long_tp_1, max(long_tp_2, long_tp_3))
                long_entry_3_hit = True

            # Long Take Profit Exits
            if (l1 or l2 or l3) and (mid_prices.vbt.crossed_above(current_long_take_profit_value).iloc[-1].item()):                
                # Long Close Trade Happens Here
                print(f"Closing All Long Positions (Take Profit): {latest_msg.instrument} Price: {mid_price}")
                close_all_positions()
                l1 = False
                l2 = False
                l3 = False
                long_tp_hit = True
                short_cond = False
                long_cond = True

            # Long Stop Loss Exit    
            if (l1 or l2 or l3) and (mid_prices.vbt.crossed_below(current_long_stop_loss_value).iloc[-1].item()):
                # Long Close Trade Happens Here
                print(f"Closing All Long Positions (Stop-Loss): {latest_msg.instrument} Price: {mid_price}")
                close_all_positions()                   
                l1 = False
                l2 = False
                l3 = False
                long_sl_hit = True
                short_cond = True
                long_cond = False
                
            ######################## SHORTS ############################

            # Short Entries
            if short_entry_1_default_cond and \
            short_cond and \
            (not s1) and \
            (not (l1 or l2 or l3)) and \
            (not np.isnan(lps_short_sl_atr_value)) and \
            (not np.isnan(lps_short_take_profit_value)):      
                s1 = True # Set Entry 1 Position to True
                # Short Entry 1 Trade Happens Here
                print(f"Short Entry 1 Executed: {latest_msg.instrument} Price: {mid_price}")
                execute_order(instrument, 0.33, is_long=False)                
                short_stop_1 = lps_short_sl_atr_value # Set Stop Loss 1 to ATR Stop Loss Value                
                # Set Stop Loss Value to the max of the 3 stop loss values
                current_short_stop_loss_value = max(short_stop_1, max(short_stop_2, short_stop_3))
                short_tp_1 = lps_short_take_profit_value       
                current_short_take_profit_value = min(short_tp_1, min(short_tp_2, short_tp_3))
                short_entry_1_hit = True
                
            if short_entry_2_default_cond and \
            short_cond and \
            (not s2) and \
            (not (l1 or l2 or l3)) and \
            (not np.isnan(lps_short_sl_atr_value)) and \
            (not np.isnan(lps_short_take_profit_value)):    
                s2 = True
                # Short Entry 2 Trade Happens Here
                print(f"Short Entry 2 Executed: {latest_msg.instrument} Price: {mid_price}")
                execute_order(instrument, 0.33, is_long=False)                
                short_stop_2 = lps_short_sl_atr_value       
                current_short_stop_loss_value = max(short_stop_1, max(short_stop_2, short_stop_3))
                short_tp_2 = lps_short_take_profit_value       
                current_short_take_profit_value = min(short_tp_1, min(short_tp_2, short_tp_3))
                short_entry_2_hit = True
                
            if short_entry_3_default_cond and \
            short_cond and \
            (not s3) and \
            (not (l1 or l2 or l3)) and \
            (not np.isnan(lps_short_sl_atr_value)) and \
            (not np.isnan(lps_short_take_profit_value)):       
                s3 = True
                # Short Entry 3 Trade Happens Here
                print(f"Short Entry 3 Executed: {latest_msg.instrument} Price: {mid_price}")
                execute_order(instrument, 0.33, is_long=False)                
                short_stop_3 = lps_short_sl_atr_value       
                current_short_stop_loss_value = max(short_stop_1, max(short_stop_2, short_stop_3))
                short_tp_3 = lps_short_take_profit_value       
                current_short_take_profit_value = min(short_tp_1, min(short_tp_2, short_tp_3))
                short_entry_3_hit = True
                
            # Short Take Profit Exit
            if (s1 or s2 or s3) and (mid_prices.vbt.crossed_below(current_short_take_profit_value).iloc[-1].item()):                
                # Short Close Trade Happens Here
                print(f"Closing All Short Positions (Take Profit): {latest_msg.instrument} Price: {mid_price}")
                close_all_positions()
                s1 = False
                s2 = False
                s3 = False
                short_tp_hit = True
                short_cond = True
                long_cond = False

            # Short Stop Loss Exit
            if (s1 or s2 or s3) and (mid_prices.vbt.crossed_above(current_short_stop_loss_value).iloc[-1].item()):                    
                # Short Close Trade Happens Here
                print(f"Closing All Short Positions (Stop-Loss): {latest_msg.instrument} Price: {mid_price}")
                close_all_positions()           
                s1 = False
                s2 = False
                s3 = False
                short_sl_hit = True
                short_cond = False
                long_cond = True

            # Update Dash App Chart:
            with lock:
                # Update Redis data
                data = decode_data(r)
                
            df = pd.DataFrame({
                'Open time': [LTF.index[-1]], # x-axis
                'Open': [LTF_open.iloc[-1].item()], # 1-minute candle open
                'High': [LTF_high.iloc[-1].item()], # 1-minute candle high
                'Low': [LTF_low.iloc[-1].item()], # 1-minute candle low
                'Close': [LTF_close.iloc[-1].item()], # 1-minute candle close
                'short cond': [short_cond], # Short condition
                'long cond' : [long_cond], # Long condition
                's1' : [s1], # short entry 1 active
                's2' : [s2], # short entry 2 active
                's3' : [s3], # short entry 3 active
                'l1' : [l1], # long entry 1 active
                'l2' : [l2], # long entry 2 active
                'l3' : [l3], # long entry 3 active
                # short entry 1 scatter plot - True/False on whether date index should have a mark. (white)
                'short entry 1': [short_entry_1_hit],
                # short entry 2 scatter plot - True/False on whether date index should have a mark. (yellow)
                'short entry 2': [short_entry_2_hit], 
                # short entry 3 scatter plot - True/False on whether date index should have a mark. (green)
                'short entry 3': [short_entry_3_hit], 
                # same as above but for long entries.
                'long entry 1': [long_entry_1_hit],
                'long entry 2': [long_entry_2_hit],
                'long entry 3': [long_entry_3_hit],
                # short stop-loss hit scatter plot - True/False on whether date index should have a mark. (red)
                'short stop-loss hit': [short_sl_hit], 
                'long stop-loss hit': [long_sl_hit], # same as above, long stop-loss hit scatter plot (red)
                'short take-profit hit': [short_tp_hit], # same as above, short take-profit hit scatter plot (lightblue)
                'long take-profit hit': [long_tp_hit], # same as above, long take-profit hit scatter plot (lightblue)
                'bids': [latest_msg.bids[0].price], # bid price
                'asks': [latest_msg.asks[0].price], # ask price
                # line plot (white) and y-axis level for short entry 1
                'short entry 1 level': [lps_short_entry_min_value], 
                # line plot (yellow) and y-axis level for short entry 2
                'short entry 2 level': [lps_short_entry_mid_value],
                # line plot (green) and y-axis level for short entry 3
                'short entry 3 level': [lps_short_entry_max_value],
                # line plot (white) and y-axis level for long entry 1
                'long entry 1 level': [lps_long_entry_max_value],
                # line plot (yellow) and y-axis level for long entry 2
                'long entry 2 level': [lps_long_entry_mid_value],
                # line plot (green) and y-axis level for long entry 3
                'long entry 3 level': [lps_long_entry_min_value],
                # line plot (red) and y-axis level for short stop-loss hit - only show (s1 or s2 or s3) == True
                'short stop-loss': [current_short_stop_loss_value],
                # line plot (red) and y-axis level for long stop-loss hit - only show (l1 or l2 or l3) == True
                'long stop-loss': [current_long_stop_loss_value],
                # line plot (lightblue) and y-axis level for short take-profit hit - only show (s1 or s2 or s3) == True
                'short take-profit': [current_short_take_profit_value],
                # line plot (lightblue) and y-axis level for long take-profit hit - only show (l1 or l2 or l3) == True
                'long take-profit': [current_long_take_profit_value],
            })
            
            data = pd.concat([data, df], ignore_index=True)
                
            # Convert all Timestamp objects to strings
            data = data.applymap(lambda x: x.strftime("%Y-%m-%dT%H:%M:%S.%f%z") if isinstance(x, pd.Timestamp) else x)

            with lock:
                # Save data back into Redis
                r.set('data', json.dumps(data.to_dict('records')))

            # Revert the plotting triggers to False
            long_tp_hit = False
            long_sl_hit = False
            short_tp_hit = False
            short_sl_hit = False
            long_entry_1_hit = False
            long_entry_2_hit = False
            long_entry_3_hit = False
            short_entry_1_hit = False
            short_entry_2_hit = False
            short_entry_3_hit = False
            
            # Add a sleep interval to avoid consuming too much CPU
            time.sleep(0.1)  # Sleep for 100ms  
            
# Create and start the streaming thread
stream_thread = threading.Thread(target=streaming_task)
stream_thread.start()  
        
# Start the main_loop thread
main_loop = threading.Thread(target=main_loop)
main_loop.start()
            
if __name__ == '__main__':    
    app.run_server(host='0.0.0.0', port=8050, debug=True, use_reloader=False) # Run the Dash app