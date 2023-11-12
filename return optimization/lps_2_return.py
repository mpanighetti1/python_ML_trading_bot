######################## Imports #############################

# Python Imports
import time
from tqdm import tqdm
import warnings
import gc
import json
import sys
import random
import math

import pandas as pd
import numpy as np
import datetime
from datetime import timedelta
from collections import namedtuple

# Numba Imports
from numba import njit
from numba import prange

# Machine Learning Imports
import optuna as opt

# Trading Imports
import talib
import vectorbtpro as vbt

##################### Data Gathering #########################

# Define column names
column_names = ['Gmt time', 'Open', 'High', 'Low', 'Close', 'Volume']

# Fetch Locally Saved Data (1 minute)
m1_data = pd.read_csv('data/forexsb/EURUSD1.csv', 
                      delimiter='\t',  # Data is comma-separated
                      names=column_names)

# Remove unwanted characters
m1_data['Gmt time'] = m1_data['Gmt time'].str.replace('\t', '')

# Convert 'Gmt time' to datetime
m1_data['Gmt time'] = pd.to_datetime(m1_data['Gmt time'], format='%Y-%m-%d %H:%M')

# Set 'Gmt time' as the index
m1_data.set_index('Gmt time', inplace=True)

# Ensure index is a DatetimeIndex
m1_data.index = pd.DatetimeIndex(m1_data.index.values)

# Rename index
m1_data.index.name = 'Open time'

# Set data for vectorbt
m1_data = vbt.Data.from_data({'EURUSD': m1_data})

## 1m data
m1_open_raw  = m1_data.get('Open')
m1_high_raw  = m1_data.get('High')
m1_low_raw   = m1_data.get('Low')
m1_close_raw = m1_data.get('Close')

# Assuming m1_data is a DataFrame with the raw data
m1_data = pd.DataFrame({
    'Open': m1_open_raw,
    'High': m1_high_raw,
    'Low': m1_low_raw,
    'Close': m1_close_raw
})

# Resample to 1-minute frequency, using NaN for missing values
m1_data_resampled = m1_data.resample('1T').asfreq()

# Drop any rows with NaN values
m1_data_resampled.dropna(inplace=True)

# If you want to reset the index to continuous minutes from the beginning point
new_index = pd.date_range(start=m1_data_resampled.index[0], periods=len(m1_data_resampled), freq='1T')
m1_data_resampled.set_index(new_index, inplace=True)
m1_data_resampled.index.name = 'Open time'

# Extracting the individual columns
m1_open = m1_data_resampled['Open']
m1_high = m1_data_resampled['High']
m1_low = m1_data_resampled['Low']
m1_close = m1_data_resampled['Close']

#              0     1      2      3      4      5       6
timeframes = ['1T', '5T', '15T', '30T', '60T', '120T', '240T']

# end value is 4, so 'LTF' can be at most '60T'
LTF_index = 0 #trial.suggest_int('LTF_index', 0, 6)
# 'HTF' is at least one step higher than 'LTF'
HTF_index = 1 #trial.suggest_int('HTF_index', 0, 6)  

LTF = timeframes[LTF_index]
HTF = timeframes[HTF_index]

# Continuous indexing for LTF data
LTF_open = m1_open.resample(LTF).first()
LTF_high = m1_high.resample(LTF).max()
LTF_low = m1_low.resample(LTF).min()
LTF_close = m1_close.resample(LTF).last()

# Continuous indexing for HTF data
HTF_open = m1_open.resample(HTF).first()
HTF_high = m1_high.resample(HTF).max()
HTF_low = m1_low.resample(HTF).min()
HTF_close = m1_close.resample(HTF).last()

##################### Ensure 2-Dim Arrays #######################

def ensure_2d(arr):
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr

######################### Initial Parameters ############################

# Starting Testing Parameters:
testing_stage = 2
saved_objective_value = None

# Core
saved_long_stop_period = 40 # Locked
saved_long_stop_multiplier = 10.0 # Locked
saved_short_stop_period = 20 # Locked
saved_short_stop_multiplier = 35.0 # Locked
saved_target_lookback_input = 20 # Locked
saved_rsi_entry_lower_threshold = 96
saved_rsi_period = 20
saved_rsi_entry_upper_threshold = 56
saved_ma_fast_period = 230
saved_ma_mid_period = 260
saved_ma_slow_period = 270
saved_short_target_multiple = 0.0
saved_long_target_multiple = 0.0
saved_short_entry_multiple = 0.0
saved_long_entry_multiple = 0.0
saved_will_frac_period = 5

# Create a mapping of parameter names to their respective global variable names
PARAM_TO_GLOBAL_VAR_MAP = {
    'long_stop_period': 'saved_long_stop_period',
    'long_stop_multiplier': 'saved_long_stop_multiplier',
    'short_stop_period': 'saved_short_stop_period',
    'short_stop_multiplier': 'saved_short_stop_multiplier',
    'target_lookback_input': 'saved_target_lookback_input',
    'rsi_entry_lower_threshold': 'saved_rsi_entry_lower_threshold',
    'rsi_period': 'saved_rsi_period',
    'rsi_entry_upper_threshold': 'saved_rsi_entry_upper_threshold',
    'ma_fast_period': 'saved_ma_fast_period',
    'ma_mid_period': 'saved_ma_mid_period',
    'ma_slow_period': 'saved_ma_slow_period',
    'short_target_multiple': 'saved_short_target_multiple',
    'long_target_multiple': 'saved_long_target_multiple',
    'short_entry_multiple': 'saved_short_entry_multiple',
    'long_entry_multiple': 'saved_long_entry_multiple',
    'will_frac_period': 'saved_will_frac_period',
    'testing_stage': 'testing_stage',
    'saved_objective_value': 'saved_objective_value'
}

# Narrow
def suggest_will_frac_period(trial):
    return ("will_frac_period", trial.suggest_int("will_frac_period", 2, 20, step=1))
def suggest_ma_fast_period(trial):
    return ("ma_fast_period", trial.suggest_int("ma_fast_period", 10, ma_mid_period - 10, step=10))
def suggest_ma_mid_period(trial):
    return ("ma_mid_period", trial.suggest_int("ma_mid_period", ma_fast_period + 10, ma_slow_period - 10, step=10))
def suggest_ma_slow_period(trial):
    return ("ma_slow_period", trial.suggest_int("ma_slow_period", ma_mid_period + 10, 2000, step=10))
def suggest_target_lookback_input(trial):
    return ("target_lookback_input", trial.suggest_int("target_lookback_input", 10, 1000, step=10))
def suggest_long_stop_period(trial):
    return ("long_stop_period", trial.suggest_int("long_stop_period", 20, 500, step=20))
def suggest_short_stop_period(trial):
    return ("short_stop_period", trial.suggest_int("short_stop_period", 20, 500, step=20))
def suggest_long_stop_multiplier(trial):
    return ("long_stop_multiplier", trial.suggest_float("long_stop_multiplier", 2.0, 100.0, step=1.0))
def suggest_short_stop_multiplier(trial):
    return ("short_stop_multiplier", trial.suggest_float("short_stop_multiplier", 2.0, 100.0, step=1.0))
def suggest_rsi_entry_lower_threshold(trial):
    return ("rsi_entry_lower_threshold", trial.suggest_int("rsi_entry_lower_threshold", 0, 100))
def suggest_rsi_entry_upper_threshold(trial):
    return ("rsi_entry_upper_threshold", trial.suggest_int("rsi_entry_upper_threshold", 0, 100))
def suggest_short_target_multiple(trial):
    return ("short_target_multiple", trial.suggest_float("short_target_multiple", 0.0, 0.001, step=0.00001))
def suggest_long_target_multiple(trial):
    return ("long_target_multiple", trial.suggest_float("long_target_multiple", 0.0, 0.001, step=0.00001))
def suggest_short_entry_multiple(trial):
    return ("short_entry_multiple", trial.suggest_float("short_entry_multiple", 0.0, 0.001, step=0.00001))
def suggest_long_entry_multiple(trial):
    return ("long_entry_multiple", trial.suggest_float("long_entry_multiple", 0.0, 0.001, step=0.00001))
def suggest_rsi_period(trial):
    return ("rsi_period", trial.suggest_int("rsi_period", 10, 1500, step=10))

##################### Indicator Functions #######################

##################### MA Indicator ###########################

@njit(nogil=True)
def get_ma_nb(ma_slow, ma_mid, ma_fast):
    # Logic for ma_long and ma_short
    ma_long = (ma_slow <= ma_mid) & (ma_mid <= ma_fast)
    ma_short = (ma_slow >= ma_mid) & (ma_mid >= ma_fast)
    return ma_long, ma_short

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
def find_highest_lowest(high, low, target_lookback_input):
    highest_of_past_length = vbt.talib('MAX').run(high, timeperiod=target_lookback_input, skipna=True).real.to_numpy().astype(np.float64)
    lowest_of_past_length = vbt.talib('MIN').run(low, timeperiod=target_lookback_input, skipna=True).real.to_numpy().astype(np.float64)
    return highest_of_past_length, lowest_of_past_length

@njit(nogil=True)
def will_frac(high, 
              low, 
              highest_of_past_length_target, 
              lowest_of_past_length_target, 
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
            long_take_profit_value[n] = highest_of_past_length_target[n] * (1 + long_target_multiple)

            if (not np.isnan(short_entry_first) and not np.isnan(short_entry_second) and not np.isnan(short_entry_third)):
                short_entry_max_value[n] = (max(short_entry_first, short_entry_second, short_entry_third))
                short_entry_min_value[n] = (min(short_entry_first, short_entry_second, short_entry_third))
                short_entry_mid_value[n] = ((short_entry_first + short_entry_second + short_entry_third - short_entry_max_value[n] - short_entry_min_value[n]))
                
                short_entry_max_value[n] = short_entry_max_value[n] * (1 + short_entry_multiple)
                short_entry_min_value[n] = short_entry_min_value[n] * (1 + short_entry_multiple)
                short_entry_mid_value[n] = short_entry_mid_value[n] * (1 + short_entry_multiple)
            
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
            short_take_profit_value[n] = lowest_of_past_length_target[n] * (1 - short_target_multiple)

            if (not np.isnan(long_entry_first) and not np.isnan(long_entry_second) and not np.isnan(long_entry_third)):
                long_entry_max_value[n] = (max(long_entry_first, long_entry_second, long_entry_third))
                long_entry_min_value[n] = (min(long_entry_first, long_entry_second, long_entry_third))
                long_entry_mid_value[n] = (long_entry_first + long_entry_second + long_entry_third - long_entry_max_value[n] - long_entry_min_value[n])
                
                long_entry_max_value[n] = long_entry_max_value[n] * (1 - long_entry_multiple)
                long_entry_min_value[n] = long_entry_min_value[n] * (1 - long_entry_multiple)
                long_entry_mid_value[n] = long_entry_mid_value[n] * (1 - long_entry_multiple)

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
    
######################### Liquidity Purge Strategy - LTF ############################

expr = """
LiquidityPurgeStrategy_LTF[lps_LTF]:

# ATR Stop Loss
x, long_stop_value = get_sl_atr(@in_LTF_high,
                                @in_LTF_low,
                                @in_LTF_close,
                                @p_long_stop_period,
                                @p_long_stop_multiplier)
                        
short_stop_value, x = get_sl_atr(@in_LTF_high,
                                 @in_LTF_low,
                                 @in_LTF_close,
                                 @p_short_stop_period,
                                 @p_short_stop_multiplier)

# Williams Fractals
highest_of_past_length_target, lowest_of_past_length_target = find_highest_lowest(@in_LTF_high, @in_LTF_low, @p_target_lookback_input)

(topFractals, 
botFractals, 
long_take_profit_value, 
short_entry_max_value, 
short_entry_min_value, 
short_entry_mid_value, 
short_take_profit_value, 
long_entry_max_value, 
long_entry_min_value, 
long_entry_mid_value) = will_frac(
                        @in_LTF_high, 
                        @in_LTF_low, 
                        highest_of_past_length_target,
                        lowest_of_past_length_target,
                        @p_will_frac_period, 
                        @p_long_target_multiple,
                        @p_short_target_multiple,
                        @p_long_entry_multiple,
                        @p_short_entry_multiple)

# Returns
topFractals, botFractals, \
long_take_profit_value, long_stop_value, short_entry_max_value, short_entry_min_value, short_entry_mid_value, \
short_take_profit_value, short_stop_value, long_entry_max_value, long_entry_min_value, long_entry_mid_value

"""

LiquidityPurgeStrategy_LTF = vbt.IF.from_expr(
    expr,
    takes_1d=True,

    # Williams Fractals
    long_stop_period=saved_long_stop_period,
    long_stop_multiplier=saved_long_stop_multiplier,
    short_stop_period=saved_short_stop_period,
    short_stop_multiplier=saved_short_stop_multiplier,
    target_lookback_input=saved_target_lookback_input,
    will_frac_period=saved_will_frac_period,
    long_target_multiple=saved_long_target_multiple,
    short_target_multiple=saved_short_target_multiple,
    long_entry_multiple=saved_long_entry_multiple,
    short_entry_multiple=saved_short_entry_multiple,
    get_sl_atr=get_sl_atr,
    find_highest_lowest=find_highest_lowest,
    will_frac=will_frac
)

####################### Multi-MA Indicator ###############################

expr = """
MultiMA[ma]:

# Value Generation
ma_fast = @talib_sma(@in_LTF_close, @p_ma_fast_period)
ma_mid = @talib_sma(@in_LTF_close, @p_ma_mid_period)
ma_slow = @talib_sma(@in_LTF_close, @p_ma_slow_period)

ma_long, ma_short = get_ma_nb(ma_slow, ma_mid, ma_fast)

# Returns
ma_slow, ma_mid, ma_fast, ma_long, ma_short

"""

MultiMA = vbt.IF.from_expr(
    expr,
    takes_1d=True,
    
    # Multi-MA Indicator
    ma_fast_period=saved_ma_fast_period,
    ma_mid_period=saved_ma_mid_period,
    ma_slow_period=saved_ma_slow_period,
    get_ma_nb=get_ma_nb
)

######################### RSI Entry Indicator ##################################

expr = """
RelativeStrengthIndex[rsi_entry]:

# RSI Indicator
rsi_entry_above_upper_threshold, rsi_entry_below_lower_threshold = get_rsi(@in_LTF_close, 
                                                                @p_rsi_period,
                                                                @p_rsi_entry_upper_threshold, 
                                                                @p_rsi_entry_lower_threshold)

# Returns
rsi_entry_above_upper_threshold, rsi_entry_below_lower_threshold

"""

RelativeStrengthIndex_entry = vbt.IF.from_expr(
    expr,
    takes_1d=True,
    
    # RSI Indicator
    rsi_period=saved_rsi_period,
    rsi_entry_upper_threshold=saved_rsi_entry_upper_threshold,
    rsi_entry_lower_threshold=saved_rsi_entry_lower_threshold,
    get_rsi=get_rsi, 
)

################## Clean Signals SuperTrend/MA Directional Bias ###################

@njit(nogil=True, parallel=True)
def clean_signals_nb(   short_entry_1_no_ma_cond_np,
                        short_entry_2_no_ma_cond_np,
                        short_entry_3_no_ma_cond_np,
                        short_entry_1_default_cond_np,
                        short_entry_2_default_cond_np,
                        short_entry_3_default_cond_np,
                        short_take_profit_value_raw,
                        short_stop_value_raw,
                        long_entry_1_no_ma_cond_np,
                        long_entry_2_no_ma_cond_np,
                        long_entry_3_no_ma_cond_np,
                        long_entry_1_default_cond_np,
                        long_entry_2_default_cond_np,
                        long_entry_3_default_cond_np,
                        long_take_profit_value_raw,
                        long_stop_value_raw,
                        m1_high_np,
                        m1_low_np,
                        m1_open_np,
                        m1_close_np):
    
    # Initialize signal arrays
    long_stop_value_np = np.full_like(m1_open_np, np.nan)
    short_stop_value_np = np.full_like(m1_open_np, np.nan)
    
    long_take_profit_value_np = np.full_like(m1_open_np, np.nan)
    short_take_profit_value_np = np.full_like(m1_open_np, np.nan)

    long_entries_1 = np.full_like(m1_open_np, False, dtype=np.bool_)
    long_entries_2 = np.full_like(m1_open_np, False, dtype=np.bool_)
    long_entries_3 = np.full_like(m1_open_np, False, dtype=np.bool_)

    short_entries_1 = np.full_like(m1_open_np, False, dtype=np.bool_)
    short_entries_2 = np.full_like(m1_open_np, False, dtype=np.bool_)
    short_entries_3 = np.full_like(m1_open_np, False, dtype=np.bool_)

    long_take_profit_hit_np = np.full_like(m1_open_np, False, dtype=np.bool_)
    short_take_profit_hit_np = np.full_like(m1_open_np, False, dtype=np.bool_)

    long_stop_hit_np = np.full_like(m1_open_np, False, dtype=np.bool_)
    short_stop_hit_np = np.full_like(m1_open_np, False, dtype=np.bool_)
    
    for col in prange(m1_open_np.shape[1]):
      
        # Column Prep
        s1 = False
        s2 = False
        s3 = False
        l1 = False
        l2 = False
        l3 = False
        short_tp_1 = 0
        short_tp_2 = 0
        short_tp_3 = 0
        long_tp_1 = 1e20
        long_tp_2 = 1e20
        long_tp_3 = 1e20
        short_stop_1 = 0
        short_stop_2 = 0
        short_stop_3 = 0
        long_stop_1 = 1e20
        long_stop_2 = 1e20
        long_stop_3 = 1e20

        for i in range(m1_open_np.shape[0]):
            
            ######################## CLEANING ############################
                
            if np.isnan(long_take_profit_value_np[i, col]): # If the value is NaN
                # Set the value to the previous value (to be overwritten if entry occurs)
                long_take_profit_value_np[i, col] = long_take_profit_value_np[i - 1, col] 
            if np.isnan(short_take_profit_value_np[i, col]):
                short_take_profit_value_np[i, col] = short_take_profit_value_np[i - 1, col]
                
            if np.isnan(long_stop_value_np[i, col]): # If the value is NaN
                # Set the value to the previous value (to be overwritten if entry occurs)
                long_stop_value_np[i, col] = long_stop_value_np[i - 1, col] 
            if np.isnan(short_stop_value_np[i, col]):
                short_stop_value_np[i, col] = short_stop_value_np[i - 1, col]

            ######################## LONGS ############################
            
            if (l1 or l2 or l3):
                long_entry_1_default_cond_np[i, col] = long_entry_1_no_ma_cond_np[i, col]
                long_entry_2_default_cond_np[i, col] = long_entry_2_no_ma_cond_np[i, col]
                long_entry_3_default_cond_np[i, col] = long_entry_3_no_ma_cond_np[i, col]

            # Long Entries
            if long_entry_1_default_cond_np[i, col] and \
            (not (s1 or s2 or s3)) and \
            (not l1) and \
            (not np.isnan(long_take_profit_value_raw[i, col])) and \
            (not np.isnan(long_stop_value_raw[i, col])):
                l1 = True # Set Entry 1 Position to True
                long_entries_1[i, col] = True # Set Entry 1 Signal to True
                
                long_tp_1 = long_take_profit_value_raw[i, col] # Set take profit 1 to value
                # Set take profit value to the min of the 3 take profit values
                long_take_profit_value_np[i, col] = min(long_tp_1, min(long_tp_2, long_tp_3))
                
                long_stop_1 = long_stop_value_raw[i, col] # Set Stop 1 to Value
                # Set stop value to the min of the 3 stop values
                long_stop_value_np[i, col] = min(long_stop_1, min(long_stop_2, long_stop_3))
                
            if long_entry_2_default_cond_np[i, col] and \
            (not (s1 or s2 or s3)) and \
            (not l2) and \
            (not np.isnan(long_take_profit_value_raw[i, col])) and \
            (not np.isnan(long_stop_value_raw[i, col])):
                l2 = True
                long_entries_2[i, col] = True
                
                long_tp_2 = long_take_profit_value_raw[i, col] # Set Take Profit 1 to Value
                # Set Take Profit Value to the min of the 3 take profit values
                long_take_profit_value_np[i, col] = min(long_tp_1, min(long_tp_2, long_tp_3))
                
                long_stop_2 = long_stop_value_raw[i, col] # Set Stop 1 to Value
                # Set stop value to the min of the 3 stop values
                long_stop_value_np[i, col] = min(long_stop_1, min(long_stop_2, long_stop_3))
                
            if long_entry_3_default_cond_np[i, col] and \
            (not (s1 or s2 or s3)) and \
            (not l3) and \
            (not np.isnan(long_take_profit_value_raw[i, col])) and \
            (not np.isnan(long_stop_value_raw[i, col])):
                l3 = True
                long_entries_3[i, col] = True
                
                long_tp_3 = long_take_profit_value_raw[i, col] # Set Take Profit 1 to Value
                # Set Take Profit Value to the min of the 3 take profit values
                long_take_profit_value_np[i, col] = min(long_tp_1, min(long_tp_2, long_tp_3))
                
                long_stop_3 = long_stop_value_raw[i, col] # Set Stop 1 to Value
                # Set stop value to the min of the 3 stop values
                long_stop_value_np[i, col] = min(long_stop_1, min(long_stop_2, long_stop_3))

            # Long Take Profit Exits
            if (l1 or l2 or l3) and (m1_high_np[i, col] >= long_take_profit_value_np[i, col]):
                long_take_profit_hit_np[i, col] = True
                l1 = False
                l2 = False
                l3 = False
                long_stop_1 = 1e20
                long_stop_2 = 1e20
                long_stop_3 = 1e20
                long_tp_1 = 1e20
                long_tp_2 = 1e20
                long_tp_3 = 1e20

            # Long Stop Loss Exit (MA Exit)
            if (l1 or l2 or l3) and (m1_low_np[i, col] <= long_stop_value_np[i, col]):
                long_stop_hit_np[i, col] = True
                l1 = False
                l2 = False
                l3 = False
                long_stop_1 = 1e20
                long_stop_2 = 1e20
                long_stop_3 = 1e20
                long_tp_1 = 1e20
                long_tp_2 = 1e20
                long_tp_3 = 1e20

                
            ######################## SHORTS ############################
            
            if (s1 or s2 or s3):
                short_entry_1_default_cond_np[i, col] = short_entry_1_no_ma_cond_np[i, col]
                short_entry_2_default_cond_np[i, col] = short_entry_2_no_ma_cond_np[i, col]
                short_entry_3_default_cond_np[i, col] = short_entry_3_no_ma_cond_np[i, col]
            
            # Short Entries
            if short_entry_1_default_cond_np[i, col] and \
            (not (l1 or l2 or l3)) and \
            (not s1) and \
            (not np.isnan(short_take_profit_value_raw[i, col])) and \
            (not np.isnan(short_stop_value_raw[i, col])):
                s1 = True # Set Entry 1 Position to True
                short_entries_1[i, col] = True # Set Entry 1 Signal to True
                
                short_tp_1 = short_take_profit_value_raw[i, col] # Set Take Profit 1 to Value
                # Set Take Profit Value to the min of the 3 take profit values
                short_take_profit_value_np[i, col] = max(short_tp_1, max(short_tp_2, short_tp_3))
                
                short_stop_1 = short_stop_value_raw[i, col] # Set Take Profit 1 to Value
                # Set Take Profit Value to the max of the 3 take profit values
                short_stop_value_np[i, col] = max(short_stop_1, max(short_stop_2, short_stop_3))
                
            if short_entry_2_default_cond_np[i, col] and \
            (not (l1 or l2 or l3)) and \
            (not s2) and \
            (not np.isnan(short_take_profit_value_raw[i, col])) and \
            (not np.isnan(short_stop_value_raw[i, col])):
                s2 = True
                short_entries_2[i, col] = True
                
                short_tp_2 = short_take_profit_value_raw[i, col] # Set Take Profit 1 to Value
                # Set Take Profit Value to the max of the 3 take profit values
                short_take_profit_value_np[i, col] = max(short_tp_1, max(short_tp_2, short_tp_3))
                
                short_stop_2 = short_stop_value_raw[i, col] # Set Take Profit 1 to Value
                # Set Take Profit Value to the max of the 3 take profit values
                short_stop_value_np[i, col] = max(short_stop_1, max(short_stop_2, short_stop_3))
                
            if short_entry_3_default_cond_np[i, col] and \
            (not (l1 or l2 or l3)) and \
            (not s3) and \
            (not np.isnan(short_take_profit_value_raw[i, col])) and \
            (not np.isnan(short_stop_value_raw[i, col])):
                s3 = True
                short_entries_3[i, col] = True
                short_tp_3 = short_take_profit_value_raw[i, col] # Set Take Profit 1 to Value
                # Set Take Profit Value to the max of the 3 take profit values
                short_take_profit_value_np[i, col] = max(short_tp_1, max(short_tp_2, short_tp_3))
                
                short_stop_3 = short_stop_value_raw[i, col] # Set Take Profit 1 to Value
                # Set Take Profit Value to the max of the 3 take profit values
                short_stop_value_np[i, col] = max(short_stop_1, max(short_stop_2, short_stop_3))

            # Short Take Profit Exit
            if (s1 or s2 or s3) and (m1_low_np[i, col] <= short_take_profit_value_np[i, col]) :
                short_take_profit_hit_np[i, col] = True
                s1 = False
                s2 = False
                s3 = False
                short_stop_1 = 0
                short_stop_2 = 0
                short_stop_3 = 0
                short_tp_1 = 0
                short_tp_2 = 0
                short_tp_3 = 0

            # Short Stop Loss Exit (MA Exit)
            if (s1 or s2 or s3) and (m1_high_np[i, col] >= short_stop_value_np[i, col]):
                short_stop_hit_np[i, col] = True
                s1 = False
                s2 = False
                s3 = False
                short_stop_1 = 0
                short_stop_2 = 0
                short_stop_3 = 0
                short_tp_1 = 0
                short_tp_2 = 0
                short_tp_3 = 0
            
    # Returns
    return long_stop_value_np, short_stop_value_np, \
           long_take_profit_value_np, short_take_profit_value_np, \
           long_entries_1, long_entries_2, long_entries_3, \
           short_entries_1, short_entries_2, short_entries_3, \
           short_stop_hit_np, long_stop_hit_np, \
           short_take_profit_hit_np, long_take_profit_hit_np
           
######################### Pipeline 6 ############################

@njit(nogil=True, parallel=True)
def pipeline_6_nb(m1_open_np,
                order_args=(), 
                init_cash=15000):
    
    # Unpacking order_args
    (long_entries_1,
    long_entries_2,
    long_entries_3,
    long_entry_min_value_m1_np,
    long_entry_mid_value_m1_np,
    long_entry_max_value_m1_np,
    long_stop_hit_np,
    long_stop_value_np,
    long_take_profit_hit_np,
    long_take_profit_value_np,
    short_entries_1,
    short_entries_2,
    short_entries_3,
    short_entry_min_value_m1_np,
    short_entry_mid_value_m1_np,
    short_entry_max_value_m1_np,
    short_stop_hit_np,
    short_stop_value_np,
    short_take_profit_hit_np,
    short_take_profit_value_np) = order_args
        
    order_records = np.empty((m1_open_np.shape), dtype=vbt.pf_enums.order_dt)
    order_counts = np.full(m1_open_np.shape[1], 0, dtype=np.int_)
    
    for col in prange(m1_open_np.shape[1]):                
        
        exec_state = vbt.pf_enums.ExecState(
            cash=float(init_cash),
            position=0.0,
            debt=0.0,
            locked_cash=0.0,
            free_cash=float(init_cash),
            val_price=np.nan,
            value=np.nan
        )

        for i in range(m1_open_np.shape[0]):  
            
            # calculate all order triggers
            order_triggers = [  # Close Opposing Position
                                ((long_entries_1[i, col] and exec_state.debt > 0), 
                                long_entry_max_value_m1_np[i, col], 0, 0, 8),
                                
                                ((long_entries_2[i, col] and exec_state.debt > 0), 
                                long_entry_mid_value_m1_np[i, col], 0, 0, 8),
                                
                                ((long_entries_3[i, col] and exec_state.debt > 0), 
                                long_entry_min_value_m1_np[i, col], 0, 0, 8),
                                
                                ((short_entries_1[i, col] and exec_state.position > 0), 
                                short_entry_min_value_m1_np[i, col], 0, 1, 8),
                                
                                ((short_entries_2[i, col] and exec_state.position > 0), 
                                short_entry_mid_value_m1_np[i, col], 0, 1, 8),
                                
                                ((short_entries_3[i, col] and exec_state.position > 0), 
                                short_entry_max_value_m1_np[i, col], 0, 1, 8),                      

                                # Open New Positions
                                (long_entries_1[i, col], long_entry_max_value_m1_np[i, col], 3000, 0, 1),
                                (long_entries_2[i, col], long_entry_mid_value_m1_np[i, col], 3000, 0, 1),
                                (long_entries_3[i, col], long_entry_min_value_m1_np[i, col], 3000, 0, 1),
                                (short_entries_1[i, col], short_entry_min_value_m1_np[i, col], 3000, 1, 1),
                                (short_entries_2[i, col], short_entry_mid_value_m1_np[i, col], 3000, 1, 1),
                                (short_entries_3[i, col], short_entry_max_value_m1_np[i, col], 3000, 1, 1),
                                
                                # Exit Current Positions @ Take Profit or Stop Loss
                                ((long_take_profit_hit_np[i, col] and exec_state.position > 0), 
                                long_take_profit_value_np[i, col], 0, 1, 8),
                                
                                ((long_stop_hit_np[i, col] and exec_state.position > 0), 
                                long_stop_value_np[i, col], 0, 1, 8),
                                
                                ((short_take_profit_hit_np[i, col] and exec_state.debt > 0), 
                                short_take_profit_value_np[i, col], 0, 0, 8),
                                
                                ((short_stop_hit_np[i, col] and exec_state.debt > 0), 
                                short_stop_value_np[i, col], 0, 0, 8)
                            ] 
                    
            val_price = m1_open_np[i, col] # Use Open Price for Portfolio Value      
            
            for trigger in order_triggers:
                
                # Unpack Trigger
                condition, price, size, direction, size_type = trigger          
                           
                # Calculate Value After Each Order
                value = exec_state.cash + val_price * exec_state.position 

                # Update Execution State
                exec_state = vbt.pf_enums.ExecState(
                    cash=exec_state.cash,
                    position=exec_state.position,
                    debt=exec_state.debt,
                    locked_cash=exec_state.locked_cash,
                    free_cash=exec_state.free_cash,
                    val_price=val_price,
                    value=value
                )
                
                if condition:
                    order = vbt.pf_nb.order_nb(
                        size=size,
                        price=price,
                        size_type=size_type,
                        direction=direction,
                        slippage=0.0000269,
                        fees=0.00005,
                    )
                else:
                    order = vbt.pf_nb.order_nothing_nb()
                    
                # Process Order Execution
                _, exec_state = vbt.pf_nb.process_order_nb(
                    col, col, i,
                    exec_state=exec_state,
                    order=order,
                    order_records=order_records,
                    order_counts=order_counts
                )
        
    return vbt.nb.repartition_nb(order_records, order_counts)

######################### Objective Function ############################

def objective(trial, LTF_high, 
                     LTF_low, 
                     LTF_close, 
                     HTF_high, 
                     HTF_low, 
                     HTF_close):

    trial.set_user_attr("constraint", (1, 1, 1, 1))
    
    ######################### Indicator Calls ###########################
    ######################### Paramaterization ##########################

    global long_stop_period, long_stop_multiplier, short_stop_period, short_stop_multiplier, \
        rsi_entry_lower_threshold, rsi_period, rsi_entry_upper_threshold, \
        target_lookback_input, \
        ma_fast_period, ma_mid_period, ma_slow_period, \
        short_target_multiple, long_target_multiple, \
        short_entry_multiple, long_entry_multiple, \
        will_frac_period

    # Default assignments
    long_stop_period = saved_long_stop_period
    long_stop_multiplier = saved_long_stop_multiplier
    short_stop_period = saved_short_stop_period
    short_stop_multiplier = saved_short_stop_multiplier
    rsi_entry_lower_threshold = saved_rsi_entry_lower_threshold
    rsi_period = saved_rsi_period
    rsi_entry_upper_threshold = saved_rsi_entry_upper_threshold
    target_lookback_input = saved_target_lookback_input
    long_target_multiple = saved_long_target_multiple
    short_target_multiple = saved_short_target_multiple
    long_entry_multiple = saved_long_entry_multiple
    short_entry_multiple = saved_short_entry_multiple
    will_frac_period = saved_will_frac_period
    ma_fast_period = saved_ma_fast_period
    ma_mid_period = saved_ma_mid_period
    ma_slow_period = saved_ma_slow_period

    if str(testing_stage).endswith(('1','4','7','0')): 
        for i in range(4):
            name, variable = chosen_functions_all3[i](trial)
            globals()[name] = variable
    elif str(testing_stage).endswith(('2','5','8')):
        for i in range(3):
            name, variable = chosen_functions_all2[i](trial)
            globals()[name] = variable
    elif str(testing_stage).endswith(('3','6','9')):
        for i in range(4):
            name, variable = chosen_functions_all3[i](trial)
            globals()[name] = variable

    lps_LTF = LiquidityPurgeStrategy_LTF.run(LTF_high, LTF_low, LTF_close, # calls LTF_high, LTF_low, LTF_close
                                        long_stop_period=long_stop_period,
                                        long_stop_multiplier=long_stop_multiplier,
                                        short_stop_period=short_stop_period,
                                        short_stop_multiplier=short_stop_multiplier,
                                        target_lookback_input=target_lookback_input,
                                        will_frac_period=will_frac_period,
                                        long_target_multiple=long_target_multiple,
                                        short_target_multiple=short_target_multiple,
                                        long_entry_multiple=long_entry_multiple,
                                        short_entry_multiple=short_entry_multiple,
                                        execute_kwargs=dict(
                                            engine='dask',
                                            chunk_len='auto',
                                            show_progress=False
                                            ))

    ma = MultiMA.run(LTF_close, # calls LTF_close
                        ma_fast_period=ma_fast_period,
                        ma_mid_period=ma_mid_period,
                        ma_slow_period=ma_slow_period,
                        execute_kwargs=dict(
                                        engine='dask',
                                        chunk_len='auto',
                                        show_progress=False
                                        ))

    rsi_entry = RelativeStrengthIndex_entry.run(LTF_close, # calls LTF_close
                        rsi_period=rsi_period,
                        rsi_entry_upper_threshold=rsi_entry_upper_threshold,
                        rsi_entry_lower_threshold=rsi_entry_lower_threshold,
                        execute_kwargs=dict(
                                        engine='dask',
                                        chunk_len='auto',
                                        show_progress=False
                                        ))

    ######################## Resample Everything To 1 Min ##########################

    # Use some placeholder value
    placeholder = -99999

    #################### Nan to Placeholder #########################

    rsi_entry_above_upper_threshold = rsi_entry.rsi_entry_above_upper_threshold.shift(1).fillna(placeholder)
    rsi_entry_below_lower_threshold = rsi_entry.rsi_entry_below_lower_threshold.shift(1).fillna(placeholder)

    ma_long = ma.ma_long.shift(1).fillna(placeholder)
    ma_short = ma.ma_short.shift(1).fillna(placeholder)
    ma_fast = ma.ma_fast.shift(1).fillna(placeholder)
    ma_mid = ma.ma_mid.shift(1).fillna(placeholder)
    ma_slow = ma.ma_slow.shift(1).fillna(placeholder)

    lps_short_entry_min_value = lps_LTF.short_entry_min_value.shift(1).fillna(placeholder)
    lps_short_entry_mid_value = lps_LTF.short_entry_mid_value.shift(1).fillna(placeholder)
    lps_short_entry_max_value = lps_LTF.short_entry_max_value.shift(1).fillna(placeholder)

    lps_long_entry_min_value = lps_LTF.long_entry_min_value.shift(1).fillna(placeholder)
    lps_long_entry_mid_value = lps_LTF.long_entry_mid_value.shift(1).fillna(placeholder)
    lps_long_entry_max_value = lps_LTF.long_entry_max_value.shift(1).fillna(placeholder)

    lps_short_take_profit_value = lps_LTF.short_take_profit_value.shift(1).fillna(placeholder)
    lps_long_take_profit_value = lps_LTF.long_take_profit_value.shift(1).fillna(placeholder)
    
    lps_short_stop_value = lps_LTF.short_stop_value.shift(1).fillna(placeholder)
    lps_long_stop_value = lps_LTF.long_stop_value.shift(1).fillna(placeholder)

    #################### Reindex To 1 Min ##########################

    ma_long_m1 = ma_long.reindex(m1_open.index).ffill()
    ma_long_m1 = ma_long_m1.bfill()
    ma_short_m1 = ma_short.reindex(m1_open.index).ffill()
    ma_short_m1 = ma_short_m1.bfill()
    ma_fast_m1 = ma_fast.reindex(m1_open.index).ffill()
    ma_mid_m1 = ma_mid.reindex(m1_open.index).ffill()
    ma_slow_m1 = ma_slow.reindex(m1_open.index).ffill()

    rsi_entry_above_upper_threshold_m1 = rsi_entry_above_upper_threshold.reindex(LTF_close.index).ffill()
    rsi_entry_above_upper_threshold_m1 = rsi_entry_above_upper_threshold_m1.bfill()
    rsi_entry_below_lower_threshold_m1 = rsi_entry_below_lower_threshold.reindex(LTF_close.index).ffill()
    rsi_entry_below_lower_threshold_m1 = rsi_entry_below_lower_threshold_m1.bfill()

    lps_short_entry_min_value_m1 = lps_short_entry_min_value.reindex(m1_open.index).ffill()
    lps_short_entry_mid_value_m1 = lps_short_entry_mid_value.reindex(m1_open.index).ffill()
    lps_short_entry_max_value_m1 = lps_short_entry_max_value.reindex(m1_open.index).ffill()

    lps_long_entry_min_value_m1 = lps_long_entry_min_value.reindex(m1_open.index).ffill()
    lps_long_entry_mid_value_m1 = lps_long_entry_mid_value.reindex(m1_open.index).ffill()
    lps_long_entry_max_value_m1 = lps_long_entry_max_value.reindex(m1_open.index).ffill()

    lps_short_take_profit_value_m1 = lps_short_take_profit_value.reindex(m1_open.index).ffill()
    lps_long_take_profit_value_m1 = lps_long_take_profit_value.reindex(m1_open.index).ffill()
    
    lps_short_stop_value_m1 = lps_short_stop_value.reindex(m1_open.index).ffill()
    lps_long_stop_value_m1 = lps_long_stop_value.reindex(m1_open.index).ffill()

    #################### Placeholder To NaN #########################

    ma_long_m1 = ma_long_m1.replace(placeholder, np.nan)
    ma_short_m1 = ma_short_m1.replace(placeholder, np.nan)
    ma_fast_m1 = ma_fast_m1.replace(placeholder, np.nan)
    ma_mid_m1 = ma_mid_m1.replace(placeholder, np.nan)
    ma_slow_m1 = ma_slow_m1.replace(placeholder, np.nan)

    rsi_entry_above_upper_threshold_m1 = rsi_entry_above_upper_threshold_m1.replace(placeholder, np.nan)
    rsi_entry_below_lower_threshold_m1 = rsi_entry_below_lower_threshold_m1.replace(placeholder, np.nan)

    short_entry_min_value_m1 = lps_short_entry_min_value_m1.replace(placeholder, np.nan)
    short_entry_mid_value_m1 = lps_short_entry_mid_value_m1.replace(placeholder, np.nan)
    short_entry_max_value_m1 = lps_short_entry_max_value_m1.replace(placeholder, np.nan)

    long_entry_min_value_m1 = lps_long_entry_min_value_m1.replace(placeholder, np.nan)
    long_entry_mid_value_m1 = lps_long_entry_mid_value_m1.replace(placeholder, np.nan)
    long_entry_max_value_m1 = lps_long_entry_max_value_m1.replace(placeholder, np.nan)

    short_take_profit_value_m1 = lps_short_take_profit_value_m1.replace(placeholder, np.nan)
    long_take_profit_value_m1 = lps_long_take_profit_value_m1.replace(placeholder, np.nan)
    
    short_stop_value_m1 = lps_short_stop_value_m1.replace(placeholder, np.nan)
    long_stop_value_m1 = lps_long_stop_value_m1.replace(placeholder, np.nan)

    ################### Entry Conditions ##########################

    # Short Entries
    short_ma_cond = ma_short_m1.astype(bool)
    short_rsi_entry_cond = rsi_entry_above_upper_threshold_m1.astype(bool)
    short_combined_entry_cond = short_ma_cond & short_rsi_entry_cond

    short_entry_1_default_cond = LTF_high.vbt.crossed_above(short_entry_min_value_m1) & short_combined_entry_cond
    short_entry_2_default_cond = LTF_high.vbt.crossed_above(short_entry_mid_value_m1) & short_combined_entry_cond
    short_entry_3_default_cond = LTF_high.vbt.crossed_above(short_entry_max_value_m1) & short_combined_entry_cond

    short_entry_1_no_ma_cond = LTF_high.vbt.crossed_above(short_entry_min_value_m1) & short_rsi_entry_cond
    short_entry_2_no_ma_cond = LTF_high.vbt.crossed_above(short_entry_mid_value_m1) & short_rsi_entry_cond
    short_entry_3_no_ma_cond = LTF_high.vbt.crossed_above(short_entry_max_value_m1) & short_rsi_entry_cond

    # Long Entries
    long_ma_cond = ma_long_m1.astype(bool)
    long_rsi_entry_cond = rsi_entry_below_lower_threshold_m1.astype(bool)
    long_combined_entry_cond = long_ma_cond & long_rsi_entry_cond

    long_entry_1_default_cond = LTF_low.vbt.crossed_below(long_entry_max_value_m1) & long_combined_entry_cond
    long_entry_2_default_cond = LTF_low.vbt.crossed_below(long_entry_mid_value_m1) & long_combined_entry_cond
    long_entry_3_default_cond = LTF_low.vbt.crossed_below(long_entry_min_value_m1) & long_combined_entry_cond

    long_entry_1_no_ma_cond = LTF_low.vbt.crossed_below(long_entry_max_value_m1) & long_rsi_entry_cond
    long_entry_2_no_ma_cond = LTF_low.vbt.crossed_below(long_entry_mid_value_m1) & long_rsi_entry_cond
    long_entry_3_no_ma_cond = LTF_low.vbt.crossed_below(long_entry_min_value_m1) & long_rsi_entry_cond

    #################### Short Values ############################

    short_entry_1_no_ma_cond_np = short_entry_1_no_ma_cond.to_numpy()
    short_entry_2_no_ma_cond_np = short_entry_2_no_ma_cond.to_numpy()
    short_entry_3_no_ma_cond_np = short_entry_3_no_ma_cond.to_numpy()
    short_entry_1_default_cond_np = short_entry_1_default_cond.to_numpy()
    short_entry_2_default_cond_np = short_entry_2_default_cond.to_numpy()
    short_entry_3_default_cond_np = short_entry_3_default_cond.to_numpy()
    short_entry_min_value_m1_np = short_entry_min_value_m1.to_numpy()
    short_entry_mid_value_m1_np = short_entry_mid_value_m1.to_numpy()
    short_entry_max_value_m1_np = short_entry_max_value_m1.to_numpy()
    short_take_profit_value_raw = short_take_profit_value_m1.to_numpy()
    short_stop_value_raw = short_stop_value_m1.to_numpy()

    #################### Long Values ############################

    long_entry_1_no_ma_cond_np = long_entry_1_no_ma_cond.to_numpy()
    long_entry_2_no_ma_cond_np = long_entry_2_no_ma_cond.to_numpy()
    long_entry_3_no_ma_cond_np = long_entry_3_no_ma_cond.to_numpy()
    long_entry_1_default_cond_np = long_entry_1_default_cond.to_numpy()
    long_entry_2_default_cond_np = long_entry_2_default_cond.to_numpy()
    long_entry_3_default_cond_np = long_entry_3_default_cond.to_numpy()
    long_entry_min_value_m1_np = long_entry_min_value_m1.to_numpy()
    long_entry_mid_value_m1_np = long_entry_mid_value_m1.to_numpy()
    long_entry_max_value_m1_np = long_entry_max_value_m1.to_numpy()
    long_take_profit_value_raw = long_take_profit_value_m1.to_numpy()
    long_stop_value_raw = long_stop_value_m1.to_numpy()

    #################### 1 Minute OHLC Data #######################

    LTF_open_np = LTF_open.to_numpy()
    LTF_high_np = LTF_high.to_numpy()
    LTF_low_np = LTF_low.to_numpy()
    LTF_close_np = LTF_close.to_numpy()

    ################### Ensure 2-Dim Array ############################

    short_entry_1_no_ma_cond_np = ensure_2d(short_entry_1_no_ma_cond_np)
    short_entry_2_no_ma_cond_np = ensure_2d(short_entry_2_no_ma_cond_np)
    short_entry_3_no_ma_cond_np = ensure_2d(short_entry_3_no_ma_cond_np)
    short_entry_1_default_cond_np = ensure_2d(short_entry_1_default_cond_np)
    short_entry_2_default_cond_np = ensure_2d(short_entry_2_default_cond_np)
    short_entry_3_default_cond_np = ensure_2d(short_entry_3_default_cond_np)
    short_take_profit_value_raw = ensure_2d(short_take_profit_value_raw)
    short_stop_value_raw = ensure_2d(short_stop_value_raw)

    long_entry_1_no_ma_cond_np = ensure_2d(long_entry_1_no_ma_cond_np)
    long_entry_2_no_ma_cond_np = ensure_2d(long_entry_2_no_ma_cond_np)
    long_entry_3_no_ma_cond_np = ensure_2d(long_entry_3_no_ma_cond_np)
    long_entry_1_default_cond_np = ensure_2d(long_entry_1_default_cond_np)
    long_entry_2_default_cond_np = ensure_2d(long_entry_2_default_cond_np)
    long_entry_3_default_cond_np = ensure_2d(long_entry_3_default_cond_np) 
    long_take_profit_value_raw = ensure_2d(long_take_profit_value_raw)
    long_stop_value_raw = ensure_2d(long_stop_value_raw)

    LTF_open_np = ensure_2d(LTF_open_np)
    LTF_high_np = ensure_2d(LTF_high_np)
    LTF_low_np = ensure_2d(LTF_low_np)
    LTF_close_np = ensure_2d(LTF_close_np)

    #################### Clean Signals Call ############################
  
    long_stop_value_np, short_stop_value_np, \
    long_take_profit_value_np, short_take_profit_value_np, \
    long_entries_1, long_entries_2, long_entries_3, \
    short_entries_1, short_entries_2, short_entries_3, \
    short_stop_hit_np, long_stop_hit_np, \
    short_take_profit_hit_np, long_take_profit_hit_np = clean_signals_nb(   short_entry_1_no_ma_cond_np,
                                                                            short_entry_2_no_ma_cond_np,
                                                                            short_entry_3_no_ma_cond_np,
                                                                            short_entry_1_default_cond_np,
                                                                            short_entry_2_default_cond_np,
                                                                            short_entry_3_default_cond_np,
                                                                            short_take_profit_value_raw,
                                                                            short_stop_value_raw,
                                                                            long_entry_1_no_ma_cond_np,
                                                                            long_entry_2_no_ma_cond_np,
                                                                            long_entry_3_no_ma_cond_np,
                                                                            long_entry_1_default_cond_np,
                                                                            long_entry_2_default_cond_np,
                                                                            long_entry_3_default_cond_np,
                                                                            long_take_profit_value_raw,
                                                                            long_stop_value_raw,
                                                                            LTF_high_np,
                                                                            LTF_low_np,
                                                                            LTF_open_np,
                                                                            LTF_close_np)
    
    ################### Ensure 2-Dim Array ############################

    long_entries_1 = ensure_2d(long_entries_1)
    long_entries_2 = ensure_2d(long_entries_2)
    long_entries_3 = ensure_2d(long_entries_3)

    long_entry_min_value_m1_np = ensure_2d(long_entry_min_value_m1_np)
    long_entry_mid_value_m1_np = ensure_2d(long_entry_mid_value_m1_np)
    long_entry_max_value_m1_np = ensure_2d(long_entry_max_value_m1_np)

    long_stop_hit_np = ensure_2d(long_stop_hit_np) 
    long_stop_value_np = ensure_2d(long_stop_value_np)

    long_take_profit_hit_np = ensure_2d(long_take_profit_hit_np) 
    long_take_profit_value_np = ensure_2d(long_take_profit_value_np)

    short_entries_1 = ensure_2d(short_entries_1)
    short_entries_2 = ensure_2d(short_entries_2)
    short_entries_3 = ensure_2d(short_entries_3)

    short_entry_min_value_m1_np = ensure_2d(short_entry_min_value_m1_np)
    short_entry_mid_value_m1_np = ensure_2d(short_entry_mid_value_m1_np)
    short_entry_max_value_m1_np = ensure_2d(short_entry_max_value_m1_np)

    short_stop_hit_np = ensure_2d(short_stop_hit_np)
    short_stop_value_np = ensure_2d(short_stop_value_np)

    short_take_profit_hit_np = ensure_2d(short_take_profit_hit_np) 
    short_take_profit_value_np = ensure_2d(short_take_profit_value_np)
    
    #################### Pipeline Call ############################

    order_records = pipeline_6_nb(
        LTF_open_np,
        order_args=(
            long_entries_1,
            long_entries_2,
            long_entries_3,
            long_entry_min_value_m1_np,
            long_entry_mid_value_m1_np,
            long_entry_max_value_m1_np,
            long_stop_hit_np,
            long_stop_value_np,
            long_take_profit_hit_np,
            long_take_profit_value_np,
            short_entries_1,
            short_entries_2,
            short_entries_3,
            short_entry_min_value_m1_np,
            short_entry_mid_value_m1_np,
            short_entry_max_value_m1_np,
            short_stop_hit_np,
            short_stop_value_np,
            short_take_profit_hit_np,
            short_take_profit_value_np
        )
    ) # Calling Main Pipeline

    ######################### Portfolio Wrapper #########################

    pf = vbt.Portfolio(
        LTF_close.vbt(freq="1m").wrapper,
        order_records,
        open=LTF_open,
        close=LTF_close,
        init_cash=15000
    )
        
    if pf.trades.count() > 0:        
        
        winning_duration = pf.trades.winning.duration.mean()
        losing_duration = pf.trades.losing.duration.mean()        

        if np.isnan(winning_duration) or np.isnan(losing_duration):
            return -10, -100, 2000
            
        avg_trade_duration = (winning_duration + losing_duration) / 2

        # Set constraints (After Lock, Constraints Are Free):

        # Account can lose 100% of its value
        max_drawdown = float(-2.0 + -(pf.max_drawdown * 100))
        # The average trade duration cannot be more than 10 hours
        filtered_trade_duration = float(0 - avg_trade_duration)
        # The number of trades cannot be less than 400
        trade_count = float(0 - pf.trades.count())
        # The total return cannot be negative
        total_return = float(0 - pf.total_return)

        trial.set_user_attr("constraint", (max_drawdown,
                                           filtered_trade_duration,
                                           trade_count,
                                           total_return))

        return pf.total_return, pf.trades.count(), avg_trade_duration

    return -10, -100, 2000
    
########################## Constraint Function #############################

def constraints(trial):
    return trial.user_attrs["constraint"]

######################### Outer Function Closure ###########################

def outer_objective(LTF_high, 
                    LTF_low, 
                    LTF_close, 
                    HTF_high, 
                    HTF_low, 
                    HTF_close, ):
    
    def call_objective(trial):
        return objective(trial, LTF_high, 
                                LTF_low, 
                                LTF_close, 
                                HTF_high, 
                                HTF_low, 
                                HTF_close, )
    return call_objective

def filter_and_sort_trials(study: opt.study.Study) -> list:
    # Get all completed trials
    completed_trials = [t for t in study.trials if t.state == opt.trial.TrialState.COMPLETE]
    # Filter trials with total_return > 0
    filtered_trials = [
        t for t in completed_trials 
        if t.values[1] >= 1
    ]
    # Sort trials based on the first objective value in descending order, [1] = count, [0] = total_return
    sorted_trials = sorted(filtered_trials, key=lambda trial: trial.values[0] if trial.values else float('-inf'), reverse=True)
    # Collect trials ensuring uniqueness by objective values
    unique_values = set()
    top_10_unique_trials = []
    for trial in sorted_trials:
        objective_values_tuple = tuple(trial.values)  # Using tuple to determine uniqueness
        if objective_values_tuple not in unique_values:
            unique_values.add(objective_values_tuple)
            top_10_unique_trials.append(trial)
        if len(top_10_unique_trials) >= 10:
            break
    return top_10_unique_trials

def get_top_trial_parameters(study: opt.study.Study) -> dict:
    top_10_unique_trials = filter_and_sort_trials(study)
    # Get the parameters of the top trial
    top_trial = top_10_unique_trials[0] if top_10_unique_trials else None
    if top_trial:
        return top_trial.params
    else:
        return {}

def get_top_trial_info(study: opt.study.Study) -> dict:
    top_10_unique_trials = filter_and_sort_trials(study)
    # Get the parameters of the top trial
    top_trial = top_10_unique_trials[0] if top_10_unique_trials else None
    if not top_trial:
        return {
            "objectives": None,
            "constraints": None
        }
    # Extract values and user attributes (i.e., constraints) from the best trial
    objective_values = top_trial.values
    constraint_values = top_trial.user_attrs.get("constraint", [])
    return {
        "objectives": objective_values,
        "constraints": constraint_values
    }

def run_optimization(testing_stage, LTF_high, 
                                    LTF_low, 
                                    LTF_close,  
                                    HTF_high, 
                                    HTF_low, 
                                    HTF_close,  startup_trials=0, n_trials=0):
    """Optimization function."""
    '''
    # Store the study in a database
    file_path = f"./tests/op_10_1_{testing_stage}.log"
    lock_obj = opt.storages.JournalFileOpenLock(file_path)
    storage = opt.storages.JournalStorage(opt.storages.JournalFileStorage(file_path, lock_obj=lock_obj),)
    '''
    # Define the SQLite database path
    sqlite_path = f"./tests/op_return_2_{testing_stage}.db"
    storage = f"sqlite:///{sqlite_path}"

    study_name = f"op_return_2_{testing_stage}"  # Use the testing_stage in study name
    warnings.filterwarnings('ignore', category=UserWarning, module='optuna.samplers._tpe.sampler')
    optimize_sampler = opt.samplers.TPESampler(n_startup_trials=startup_trials, 
                                                  constraints_func=constraints)
    # Maximize the objective value  
    study = opt.create_study(study_name=study_name,
                                storage=storage,
                                directions=['maximize', 'maximize', 'minimize'],
                                load_if_exists=True)
    study.sampler = optimize_sampler
    study.optimize(outer_objective( LTF_high, 
                                    LTF_low, 
                                    LTF_close, 
                                    HTF_high, 
                                    HTF_low, 
                                    HTF_close), 
                        n_trials=n_trials,
                        gc_after_trial=True)

    return study

def set_global_variables_from_best_params(best_params: dict):
    """Set global variable values from the best parameters."""
    for param, value in best_params.items():
        global_var_name = PARAM_TO_GLOBAL_VAR_MAP.get(param)
        if global_var_name:
            globals()[global_var_name] = value

def save_selected_variables(selected_params):
    # Load existing data
    try:
        with open('saved_variables_return_2.json', 'r') as f:
            data_to_save = json.load(f)
    except FileNotFoundError:
        data_to_save = {}

    # Update the data with the selected parameters
    for param_name in selected_params:
        global_name = PARAM_TO_GLOBAL_VAR_MAP.get(param_name)
        if global_name and global_name in globals():
            data_to_save[param_name] = globals()[global_name]

    # Save the updated data
    with open('saved_variables_return_2.json', 'w') as f:
        json.dump(data_to_save, f)

def save_to_file(filename: str, content: str) -> None:
    """Save the content to a text file with the given filename."""
    with open(filename, "w") as f:
        f.write(content)

# Load Globals From Previous Test
if len(sys.argv) > 1:
    with open(sys.argv[1], 'r') as f:
        loaded_data = json.load(f)
        # Set the loaded data to their respective global variables
        for param_name, global_name in PARAM_TO_GLOBAL_VAR_MAP.items():
            globals()[global_name] = loaded_data[param_name]

# At the end, save the important global variables
data_to_save = {}
for param_name, global_name in PARAM_TO_GLOBAL_VAR_MAP.items():
    data_to_save[param_name] = globals()[global_name]

with open('saved_variables_return_2.json', 'w') as f:
    json.dump(data_to_save, f)


suggestion_functions_all2 = [
    suggest_ma_fast_period,
    suggest_ma_slow_period,
    suggest_will_frac_period
]

suggestion_functions_all3 = [
    suggest_rsi_entry_lower_threshold,
    suggest_rsi_entry_upper_threshold,
    suggest_rsi_period,
    suggest_ma_mid_period
]

if str(testing_stage).endswith(('1','4','7','0')):
    chosen_functions_all3 = random.sample(suggestion_functions_all3, 4)
elif str(testing_stage).endswith(('2','5','8')):
    chosen_functions_all2 = random.sample(suggestion_functions_all2, 3)
elif str(testing_stage).endswith(('3','6','9')):
    chosen_functions_all3 = random.sample(suggestion_functions_all3, 4)

###########################################################################
############################### Main Program ##############################
###########################################################################

if __name__ == '__main__':
    
    ######################### Optuna Optimization #########################

    if str(testing_stage).endswith(('0','2','4','6','8')):
        n_trials = 1000 + (testing_stage * 50)
        startup_trials = (n_trials * 0.20)
    elif str(testing_stage).endswith(('1','3','5','7','9')):
        n_trials = 1000 + (testing_stage * 50)
        startup_trials = (n_trials * 0.20)

    # Store the study in a database
    study = run_optimization(testing_stage,
                             LTF_high, 
                             LTF_low, 
                             LTF_close, 
                             HTF_high, 
                             HTF_low, 
                             HTF_close, 
                             startup_trials=startup_trials, n_trials=n_trials)
    
    best_params = get_top_trial_parameters(study) # Get values from top trial
    top_trial_info = get_top_trial_info(study) # Get objective values of top trial

    # Fetch saved objective value
    saved_value = None
    try:
        with open('saved_variables_return_2.json', 'r') as f:
            saved_data = json.load(f)
            saved_value = saved_data.get('saved_objective_value')
    except FileNotFoundError:
        pass

    # if best_params isn't empty
    if best_params and top_trial_info:
        if saved_value is None or top_trial_info["objectives"][0] > saved_value:
            
            saved_objective_value = top_trial_info["objectives"][0]

            # Set globals from top trial
            set_global_variables_from_best_params(best_params) 

            testing_stage = globals().get('testing_stage', 0) + 1 # Increment

            # Save these better values to file.
            save_selected_variables(PARAM_TO_GLOBAL_VAR_MAP.keys()) 

            ####### Create Output Txt File when Objective Value Is Better ######

            content = "Recorded saved variable values:\n"
            for param_name, global_name in PARAM_TO_GLOBAL_VAR_MAP.items():
                value = globals()[global_name]
                content += f"{param_name}: {value}\n"

            ############# Record Top Trial Information #########################

            content += "\nTop trial information from the last study:\n"

            # Objective values
            content += "\nObjective values:\n"
            for idx, value in enumerate(top_trial_info["objectives"], 1):
                content += f"Objective {idx}: {value}\n"

            # Constraints
            content += "\nConstraint values:\n"
            for idx, value in enumerate(top_trial_info["constraints"], 1):
                content += f"Constraint {idx}: {value}\n"
                
            # Use the study's name as the filename
            file_name = f"./tests/results/{study.study_name}.txt"

            # Save the content to a text file
            save_to_file(file_name, content)

        else: # Save only testing_stage, do not update globals
            testing_stage = globals().get('testing_stage', 0) + 1 # Increment
            save_selected_variables(['testing_stage'])

    else: # Save only testing_stage, do not update globals
        testing_stage = globals().get('testing_stage', 0) + 1 # Increment
        save_selected_variables(['testing_stage'])

    pass