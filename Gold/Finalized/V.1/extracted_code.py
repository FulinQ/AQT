import pandas as pd
import pandas_ta as ta
from pandas.tseries.offsets import BusinessDay
import pandas_market_calendars as mcal
import pywt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import plotly.graph_objects as go
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ardl import ARDL, ardl_select_order, UECM
from statsmodels.stats.diagnostic import het_arch
from statsmodels.tools.sm_exceptions import ValueWarning
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
import fredapi as fa
from datetime import date
from twelvedata import TDClient
import vectorbt as vbt
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, LSTM, BatchNormalization, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.losses import Huber
import time
import re
import os

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore', ValueWarning)

os.chdir('/Users/fulinq/Documents/KMITL/FinancialEngineering/Y4/Y4T1/PROJECT/ARDL-ECM/Code/Gold/Finalized/V.1')

fred = fa.Fred(api_key='c948956426006ca126a2dd3bd1f07cee')
td = TDClient(apikey='aa61c51218c248698467af34d09b9d46')

def fetch_fred(fred_client, series_id, col_name, percent = False, save_csv=False):
    df = fred_client.get_series(series_id)
    df.index = pd.to_datetime(df.index) 
    print(f'NaN value before processing: {df.isna().sum()}')
    df = df.ffill()
    print(f'NaN value after processing: {df.isna().sum()}')
    df.rename(col_name, inplace=True)
    print(f'Total records for {col_name}: {len(df)}')
    print(f'start date: {df.index.min()}')
    print(f'end date: {df.index.max()}')
    
    if percent:
        df = df.mul(0.01)
        print(f'Total records for {col_name} in percent: {len(df)}')
    
    if save_csv:
        filename = f"all_{col_name.lower()}_data_fred.csv"
        df.to_csv(filename)
        print(f"FRED data saved to {filename}")
    
    return pd.DataFrame(df)

def chow_lin_disaggregate(y_low: pd.Series, X_high: pd.DataFrame,
                          agg_method: str = 'sum', rho: float = None) -> tuple:
    y_low = y_low.dropna().copy()
    X_high = X_high.dropna().copy()
    n_high_per_low = 3  # Quarterly -> Monthly = 3 เดือนต่อไตรมาส

    # หาช่วงเวลาที่ซ้อนทับกัน (Overlapping period)
    quarters = y_low.index
    months = X_high.index
    min_date = max(quarters.min(), months.min().to_period('Q').to_timestamp())
    max_date = min(quarters.max(), months.max().to_period('Q').to_timestamp())

    y_low = y_low[(y_low.index >= min_date) & (y_low.index <= max_date)]
    
    # ปรับช่วงเวลาของ Monthly ให้ครอบคลุม Quarterly พอดี
    month_start = y_low.index.min()
    month_end = (y_low.index.max() + pd.offsets.QuarterEnd()).to_period('M').to_timestamp()
    X_high = X_high[(X_high.index >= month_start) & (X_high.index <= month_end)]

    n_low = len(y_low)
    n_high = n_low * n_high_per_low
    X_high = X_high.iloc[:n_high] # ตัดส่วนเกินออก

    # Build aggregation matrix C (Matrix สำหรับแปลงรายเดือนกลับเป็นไตรมาส)
    C = np.zeros((n_low, n_high))
    for i in range(n_low):
        start_col = i * n_high_per_low
        end_col = start_col + n_high_per_low
        if agg_method == 'sum': # สำหรับ Flow variable เช่น GDP
            C[i, start_col:end_col] = 1.0
        elif agg_method == 'mean': # สำหรับ Stock variable
            C[i, start_col:end_col] = 1.0 / n_high_per_low
        else:
            C[i, end_col - 1] = 1.0

    # Prepare X matrix
    X = X_high.values
    if X.ndim == 1: X = X.reshape(-1, 1)
    X = np.column_stack([np.ones(n_high), X]) # เพิ่ม Intercept

    # OLS เบื้องต้นเพื่อหาค่า Rho (Autocorrelation coefficient)
    X_low = C @ X
    y = y_low.values.flatten()
    beta_ols = np.linalg.lstsq(X_low, y, rcond=None)[0]
    u_low = y - X_low @ beta_ols

    if rho is None: # ถ้าไม่ได้กำหนดมา ให้คำนวณจาก Residuals
        if len(u_low) > 1:
            rho = np.corrcoef(u_low[:-1], u_low[1:])[0, 1]
            rho = np.clip(rho, -0.99, 0.99)
        else:
            rho = 0.0

    # GLS Estimation (พระเอกของงาน)
    # สร้าง Covariance Matrix V ตามโครงสร้าง AR(1)
    V = np.zeros((n_high, n_high))
    for i in range(n_high):
        for j in range(n_high):
            V[i, j] = rho ** abs(i - j)

    V_low = C @ V @ C.T
    try:
        V_low_inv = np.linalg.inv(V_low)
    except:
        V_low_inv = np.linalg.pinv(V_low)

    # คำนวณ Beta ด้วย GLS
    XVX = X_low.T @ V_low_inv @ X_low
    XVy = X_low.T @ V_low_inv @ y
    try:
        beta_gls = np.linalg.solve(XVX, XVy)
    except:
        beta_gls = np.linalg.lstsq(XVX, XVy, rcond=None)[0]

    # คำนวณค่าพยากรณ์และกระจาย Error (Distribute residuals)
    p_high = X @ beta_gls
    u_low_gls = y - X_low @ beta_gls
    VCt = V @ C.T
    
    try:
        dist_matrix = VCt @ np.linalg.inv(V_low)
    except:
        dist_matrix = VCt @ np.linalg.pinv(V_low)

    y_high = p_high + dist_matrix @ u_low_gls # ผลลัพธ์สุดท้าย

    result = pd.Series(y_high, index=X_high.index, name='GDP_Monthly_ChowLin')
    return result, beta_gls, rho

gold = vbt.YFData.download("GC=F", start="2006-01-01", interval="1d").get()
gold = pd.DataFrame(gold)
gold.columns = gold.columns.str.lower()
gold.index = pd.to_datetime(gold.index).tz_localize(None)
gold = gold.sort_index()
gold = gold.drop(columns=['dividends', 'stock splits'])
gold.to_csv('all_gold_data.csv')
gold = gold.get('close')
gold

dollar_index = fetch_fred(fred, series_id='DTWEXBGS', col_name='Dollar Index')
dollar_index

ppi = fetch_fred(fred, series_id='PPIACO', col_name='PPI')
ppi

fed_fund = fetch_fred(fred, series_id='FEDFUNDS', col_name='Federal Fund Rate', percent=True)
fed_fund

vix = fetch_fred(fred, series_id='VIXCLS', percent=True,col_name='VIX')
vix['VIX'] = vix['VIX'].mul(1 / np.sqrt(252))
vix

unemploy = fetch_fred(fred, series_id='ICSA', col_name='ISCA') #Initial Claims
unemploy

ip = fetch_fred(fred, series_id='INDPRO', col_name='IP')
ip

gdp = fetch_fred(fred, series_id='GDP', col_name='GDP')
gdp

y_target = gdp['GDP']
X_indicator = ip[['IP']]

gdp_monthly_gls, beta, rho = chow_lin_disaggregate(y_low=y_target, X_high=X_indicator, agg_method='sum', rho=None)
print("Estimated Rho (Autocorrelation):", rho)
gdp = gdp_monthly_gls.copy()
gdp_monthly_gls

fed_balance = fetch_fred(fred, series_id='WALCL', col_name='Fed Balance Sheet') #Federal Reserve Total Assets
fed_balance

# 1. organize data
realtime_data = {
    'gold': gold,
    'dollar_index': dollar_index,
    'vix': vix,
    'fed_rate': fed_fund,
    'fed_balance': fed_balance,
    'labor_claims': unemploy
}

lagged_data = {
    'ip': ip,
    'gdp': gdp,
    'ppi': ppi
}

# 2. resample & rename
monthly_dfs = []

# process real-time
for name, data in realtime_data.items():
    # FIX: force rename for both Series and DataFrame to match the key (lowercase)
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0].to_frame(name)
    else:
        data = data.to_frame(name)
    
    if name in ['labor_claims', 'vix']:
        monthly_dfs.append(data.resample('ME').mean())
    else:
        monthly_dfs.append(data.resample('ME').last())

# process lagged
for name, data in lagged_data.items():
    if isinstance(data, pd.DataFrame):
        data = data.iloc[:, 0].to_frame(name)
    else:
        data = data.to_frame(name)
    monthly_dfs.append(data.resample('ME').last())

# 3. merge
df_final = pd.concat(monthly_dfs, axis=1)

# 4. handle lag (shift)
vars_to_shift = ['ip', 'ppi']
for col in vars_to_shift:
    df_final[col] = df_final[col].shift(1)
df_final['gdp'] = df_final['gdp'].shift(4)

# 5. target variable
df_final['target_gold'] = df_final['gold'].shift(-1)

# 6. feature selection
features = [
    'gold', 'dollar_index', 'vix', 'fed_rate', 
    'fed_balance', 'labor_claims', 
    'ip', 'gdp','ppi'
]

df_model = df_final[features + ['target_gold']].dropna()

# check
print(f"data range: {df_model.index.min().date()} to {df_model.index.max().date()}")
print(df_model.columns)
df_model

df_model.to_csv('gold_price_model_data.csv')

df_ret = pd.DataFrame()
cols_to_transform = ['gold', 'gdp', 'ip', 'ppi','dollar_index', 'labor_claims', 'fed_balance'] # ไม่เอา IP, PPI ตามแผน Core Model
cols_not_to_transform = ['fed_rate', 'vix'] # ตัวแปรที่ไม่ทำ log return
for col in cols_to_transform:
    if col in df_model.columns:
        df_ret[f'{col}_ret'] = np.log(df_model[col]).diff()
for col in cols_not_to_transform:
    if col in df_model.columns:
        df_ret[f'{col}_change'] = df_model[col].diff()
    
df_ret.dropna(inplace=True)
df_ret

df_model = pd.read_csv('gold_price_model_data.csv', index_col=0, parse_dates=True)
df_model

vars_to_log = ['gold', 'dollar_index', 'fed_balance', 'labor_claims', 'ip', 'gdp','ppi', 'target_gold']
for col in vars_to_log:
    df_model[f'ln_{col}'] = np.log(df_model[col])

model_vars = ['fed_rate', 'vix'] + [f'ln_{c}' for c in vars_to_log]
df_ardl = df_model[model_vars].dropna()

df_ardl

def run_adf_test(series, name):
    # Test at Level
    result = adfuller(series.dropna())
    p_value = result[1]
    
    if p_value <= 0.05:
        return f"I(0) - Stationary (p={p_value:.4f})"
    else:
        # ถ้า Level ไม่นิ่ง ให้ลอง Test แบบ Diff (First Difference)
        diff_result = adfuller(series.diff().dropna())
        diff_p_value = diff_result[1]
        
        if diff_p_value <= 0.05:
            return f"I(1) - Stationary at Diff (p={diff_p_value:.4f})"
        else:
            return f"I(2) or Higher (Non-Stationary) (p={diff_p_value:.4f})"
        
summary_data = []
for col in df_ardl.columns:
    status = run_adf_test(df_ardl[col], col)
    summary_data.append({'Variable': col, 'Status': status})

df_status = pd.DataFrame(summary_data)
df_status

X_cols = ['fed_rate'
          ,'ln_gold'
          ,'ln_dollar_index'
          ,'vix'
          ,'ln_labor_claims'
          ,'ln_ip'
        #   ,'ln_gdp'
          ,'ln_ppi'
        #   ,'ln_fed_balance'
          ]

X = df_ardl[X_cols].dropna()
X = sm.add_constant(X)

vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X.values, i) for i in range(len(X.columns))]

vif_data

train_date_str = '2015-12-31'
df_test_ardl = df_ardl[df_ardl.index <= train_date_str].copy()
df_test_ardl

y_col = 'ln_gold'
X_cols = ['fed_rate'
          ,'ln_dollar_index'
        #   ,'vix'
          # ,'ln_labor_claims'
        #   ,'ln_ip'
        #   ,'ln_gdp'
          # ,'ln_ppi'
          # ,'ln_fed_balance'
          ]

data_ardl = df_test_ardl[[y_col] + X_cols].dropna()

custom_max_order = {
    'fed_rate': 6,
    'ln_dollar_index': 6,
    # 'vix': 3,
    # 'ln_labor_claims': 6,
    # 'ln_ip': 3,
    # 'ln_ppi': 6
}

sel_res = ardl_select_order(
    data_ardl[y_col], 
    maxlag=6, 
    exog=data_ardl[X_cols], 
    maxorder=custom_max_order,
    ic='aic'
)

print(f"Best AR Lags: {sel_res.ar_lags}")
print(f"Best DL Orders: {sel_res.dl_lags}")

ar_lag = max(sel_res.ar_lags) if isinstance(sel_res.ar_lags, list) else sel_res.ar_lags
dl_lags = {k: (max(v) if isinstance(v, list) else v) for k, v in sel_res.dl_lags.items()}
exog_order = {}
for i in dl_lags:
    exog_order[i] = max(1, dl_lags[i])
    
print(f"\n--- 2. ARDL Levels Analysis & Bounds Test ---")
model_ardl = ARDL(
    data_ardl[y_col], 
    lags=ar_lag, 
    exog=data_ardl[X_cols], 
    order=exog_order
)
res_ardl = model_ardl.fit()
res_ardl.summary()

model_uecm = UECM(
    data_ardl[y_col], 
    lags=6, 
    exog=data_ardl[X_cols], 
    order=exog_order
)
res_uecm = model_uecm.fit()

# 2. รัน Bounds Test จากผลลัพธ์ของ UECM
# case 3 คือมี intercept แต่ไม่มี trend (นิยมใช้ที่สุด)
bt_results = res_uecm.bounds_test(case=3)

print("--- ARDL Bounds Test Results ---")
print(bt_results)

# 3. ดูค่า ECT (ในตาราง summary จะชื่อประมาณ 'diff.ln_gold.L1' หรือตัวแปรที่เป็นระดับ Level)
# หรือดูค่า Adjustment Term โดยตรง
print("\n--- UECM Summary (ดูค่า ECT และนัยสำคัญ) ---")
print(res_uecm.summary())

exog_order_pure = {}
for i in exog_order:
    exog_order_pure[i] = [int(j) for j in range(1, exog_order[i]+1)]
ar_order = ar_lag

train_data = df_test_ardl.copy()
test_data = df_ardl[df_ardl.index > train_date_str].copy()

history = train_data.copy()
predictions = []
actuals = test_data[y_col].values

print(f"Train Period: {train_data.index[0].date()} to {train_data.index[-1].date()} (Count: {len(train_data)})")
print(f"Test Period:  {test_data.index[0].date()} to {test_data.index[-1].date()} (Count: {len(test_data)})")
print(f"\nStarting Walk-Forward Forecast (OOS)")

for t in range(len(test_data)):
    model = ARDL(
        endog=history[y_col],
        lags=ar_order,
        exog=history[X_cols],
        order=exog_order_pure,
        trend='c'
    )
    model_fit = model.fit()
    
    next_exog = test_data.iloc[[t]][X_cols]
    
    pred = model_fit.predict(start=len(history), end=len(history), exog_oos=next_exog)
    yhat = pred.values[0]
    predictions.append(yhat)
    
    history = pd.concat([history, test_data.iloc[[t]]])
    
    # warking forward
    # history = history.iloc[1:]
    
    if (t+1) % 12 == 0:
        print(f"Step {t+1}: {test_data.index[t].date()} -> Pred={np.exp(yhat):.4f} | Actual={np.exp(actuals[t]):.4f}")

final_model = ARDL(endog=history[y_col], lags=ar_order, exog=history[X_cols], order=exog_order_pure, trend='c')
final_model_fit = final_model.fit()
next_exog_future = history.iloc[[-1]][X_cols]
pred_future = final_model_fit.predict(start=len(history), end=len(history), exog_oos=next_exog_future)
yhat_future = pred_future.values[0]
# predictions.append(yhat_future)

actual_price = np.exp(actuals)
pred_price = np.exp(predictions)

results = pd.DataFrame({
    'Actual': actuals,
    'Predicted' : predictions,
    'Error' : actuals - predictions,
    'Actual_Price': actual_price,
    'Predicted_Price': pred_price,
    'Error_Price' : actual_price - pred_price
}, index=test_data.index)
last_date = results.index[-1]
next_date = last_date + BusinessDay(n=1)
future_row = pd.DataFrame({
    'Actual': [np.nan],              
    'Predicted': [yhat_future],      
    'Error': [np.nan],               
    'Actual_Price': [np.nan],        
    'Predicted_Price': [np.exp(yhat_future)], 
    'Error_Price': [np.nan]  
}, index=[next_date]) 

results = pd.concat([results, future_row])       
results.round(2)

rmse = np.sqrt(mean_squared_error(actual_price, pred_price))
mae = mean_absolute_error(actual_price, pred_price)

print(f"RMSE (USD): {rmse:.2f}")
print(f"MAE (USD):  {mae:.2f}")

# plt.figure(figsize=(14, 7))

# plt.axvline(x=pd.to_datetime('2015-12-31'), color='gray', linestyle=':', label='Train/Test Split')

# plt.plot(df_ardl.index, np.exp(df_ardl[y_col]), label='Actual History', color='lightgray')
# plt.plot(test_data.index, actual_price, label='Actual Test Data', color='#1f77b4', linewidth=2)
# plt.plot(test_data.index, pred_price, label='Forecast (Pure OOS)', color='#d62728', linestyle='--', linewidth=2)

# plt.title('Gold Price Forecast: Out-of-Sample Testing (2016-Present)')
# plt.xlabel('Date')
# plt.ylabel('Price (USD)')
# plt.legend()
# plt.grid(True, alpha=0.3)
# plt.show()

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_ardl.index, 
    y=np.exp(df_ardl[y_col]),
    mode='lines',
    name='Actual History',
    line=dict(color='lightgray')
))

fig.add_trace(go.Scatter(
    x=test_data.index, 
    y=actual_price,
    mode='lines',
    name='Actual Test (2016-Present)',
    line=dict(color='#1f77b4', width=2)
))

fig.add_trace(go.Scatter(
    x=test_data.index, 
    y=pred_price,
    mode='lines',
    name='Forecast',
    line=dict(color='#d62728', width=2, dash='dash')
))

fig.update_layout(
    width=1000,
    height=700,
    autosize=False,
    title='Gold Price Forecast: Interactive Walk-Forward Validation',
    yaxis_title='Price (USD)',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    ),
    template="plotly_white",
    legend=dict(x=0, y=1)
)

fig.show()

exog_order_pure = {}
for i in exog_order:
    exog_order_pure[i] = [int(j) for j in range(1, exog_order[i]+1)]
ar_order = ar_lag

# --- 1. Define Stationary and Non-Stationary Variables ---
# Based on your ADF test results
i0_vars = ['vix', 'ln_labor_claims'] 
i1_vars = [col for col in df_ardl.columns if col not in i0_vars]

# --- 2. Create a Stationary DataFrame for ARDL modeling ---
df_stat = df_ardl.copy()
df_stat[i1_vars] = df_stat[i1_vars].diff() # Apply First Difference to I(1) variables ONLY
df_stat = df_stat.dropna() # Drop the first row containing NaNs from differencing

# --- 3. Split Data into Train and Test ---
# 'stat' is used for model training/predicting (differences)
train_data_stat = df_stat[df_stat.index <= train_date_str].copy()
test_data_stat  = df_stat[df_stat.index > train_date_str].copy()

# 'level' is kept for price reconstruction later
history_level = df_ardl[df_ardl.index <= train_date_str].copy()
test_data_level = df_ardl[df_ardl.index > train_date_str].copy()

history_stat = train_data_stat.copy()

predictions_diff = []
predictions_log_level = []
actuals_log_level = test_data_level[y_col].values

print(f"Train Period: {train_data_stat.index[0].date()} to {train_data_stat.index[-1].date()} (Count: {len(train_data_stat)})")
print(f"Test Period:  {test_data_stat.index[0].date()} to {test_data_stat.index[-1].date()} (Count: {len(test_data_stat)})")
print(f"\nStarting Walk-Forward Forecast (Stationary OOS)")

# --- 4. Walk-Forward Loop ---
for t in range(len(test_data_stat)):
    
    # Fit ARDL on Stationary Data
    model = ARDL(
        endog=history_stat[y_col],
        lags=ar_order,
        exog=history_stat[X_cols],
        order=exog_order_pure,
        trend='c'
    )
    model_fit = model.fit()
    
    # Predict the next difference (Return)
    next_exog = test_data_stat.iloc[[t]][X_cols]
    pred_diff = model_fit.predict(start=len(history_stat), end=len(history_stat), exog_oos=next_exog).values[0]
    
    # Reconstruct Log Level: Previous Actual Log Level + Predicted Difference
    prev_actual_log_level = history_level[y_col].iloc[-1]
    yhat_log_level = prev_actual_log_level + pred_diff
    
    predictions_diff.append(pred_diff)
    predictions_log_level.append(yhat_log_level)
    
    # Update histories with the new actual row
    history_stat = pd.concat([history_stat, test_data_stat.iloc[[t]]])
    history_level = pd.concat([history_level, test_data_level.iloc[[t]]])
    
    if (t+1) % 12 == 0:
        print(f"Step {t+1}: {test_data_stat.index[t].date()} -> Pred Price={np.exp(yhat_log_level):.2f} | Actual Price={np.exp(actuals_log_level[t]):.2f}")

# --- 5. Predict the Future (T+1) ---
final_model = ARDL(
    endog=history_stat[y_col], 
    lags=ar_order, 
    exog=history_stat[X_cols], 
    order=exog_order_pure, 
    trend='c'
)
final_model_fit = final_model.fit()

# Use the last known exog values (Naive approach) for future prediction
next_exog_future = history_stat.iloc[[-1]][X_cols] 
pred_diff_future = final_model_fit.predict(start=len(history_stat), end=len(history_stat), exog_oos=next_exog_future).values[0]

# Reconstruct future price
prev_actual_log_level_future = history_level[y_col].iloc[-1]
yhat_log_level_future = prev_actual_log_level_future + pred_diff_future

# --- 6. Assemble the Final Results DataFrame & Apply mcal ---
actual_price = np.exp(actuals_log_level)
pred_price = np.exp(predictions_log_level)

results = pd.DataFrame({
    'Actual': actuals_log_level,
    'Predicted' : predictions_log_level,
    'Error' : actuals_log_level - predictions_log_level,
    'Actual_Price': actual_price,
    'Predicted_Price': pred_price,
    'Error_Price' : actual_price - pred_price
}, index=test_data_stat.index)

last_date = results.index[-1]

# Use mcal to find the exact next CME Gold trading day
market_cal = mcal.get_calendar('CMEGlobex_Gold')
start_search = last_date + pd.Timedelta(days=1)
end_search = start_search + pd.Timedelta(days=15) # Buffer for long holidays

# Get the exact next valid trading day
valid_days = market_cal.valid_days(start_date=start_search, end_date=end_search)
next_date = valid_days[0].tz_localize(None) # Remove timezone for clean index

# Append the future prediction row
future_row = pd.DataFrame({
    'Actual': [np.nan],              
    'Predicted': [yhat_log_level_future],                    
    'Actual_Price': [np.nan],        
    'Predicted_Price': [np.exp(yhat_log_level_future)], 
    'Error': [np.nan],
    'Error_Price': [np.nan]  
}, index=[next_date]) 

results = pd.concat([results, future_row])       

print("\nFinal Results (with exact CME Trading Dates):")
results.round(2)

rmse = np.sqrt(mean_squared_error(actual_price, pred_price))
mae = mean_absolute_error(actual_price, pred_price)

print(f"RMSE (USD): {rmse:.2f}")
print(f"MAE (USD):  {mae:.2f}")

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_ardl.index, 
    y=np.exp(df_ardl[y_col]),
    mode='lines',
    name='Actual History',
    line=dict(color='lightgray')
))
fig.add_trace(go.Scatter(
    x=test_data_stat.index,
    y=actual_price,
    mode='lines',
    name='Actual Test (2016-Present)',
    line=dict(color='#1f77b4', width=2)
))
fig.add_trace(go.Scatter(
    x=test_data_stat.index,
    y=pred_price,
    mode='lines',
    name='Forecast',
    line=dict(color='#d62728', width=2, dash='dash')
))
fig.update_layout(
    width=1000,
    height=700,
    autosize=False,
    title='Gold Price Forecast: Walk-Forward Validation with Stationary ARDL',
    yaxis_title='Price (USD)',
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(count=5, label="5y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(visible=True),
        type="date"
    ),
    template="plotly_white",
    legend=dict(x=0, y=1)
)
fig.show()

plt.figure(figsize=(14, 7))

plt.axvline(x=pd.to_datetime('2015-12-31'), color='gray', linestyle=':', label='Train/Test Split')

plt.plot(df_ardl.index, np.exp(df_ardl[y_col]), label='Actual History', color='lightgray')
plt.plot(test_data.index, actual_price, label='Actual Test Data', color='#1f77b4', linewidth=2)
plt.plot(test_data.index, pred_price, label='Forecast (Pure OOS)', color='#d62728', linestyle='--', linewidth=2)

plt.title('Gold Price Forecast: Out-of-Sample Testing (2016-Present)')
plt.xlabel('Date')
plt.ylabel('Price (USD)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

def denoise_data(data, wavelet='db4', level=2):
    coeff = pywt.wavedec(data, wavelet, mode="per")
    sigma = (1/0.6745) * np.median(np.abs(coeff[-1] - np.median(coeff[-1])))
    uthesh = sigma * np.sqrt(2 * np.log(len(data)))
    new_coeff = [coeff[0]]
    for i in coeff[1:]:
        new_coeff.append(pywt.threshold(i, value=uthesh, mode='soft'))
    reconstructed = pywt.waverec(new_coeff, wavelet, mode='per')
    return reconstructed[:len(data)]

macro_feature = results[['Predicted']].copy()
macro_feature.columns = ['Macro_Signal']

# Resample monthly to daily and forward fill
macro_daily = macro_feature.resample('D').asfreq()
macro_daily = macro_daily.ffill()

# Normalize datetime index (remove time component)
macro_daily.index = pd.to_datetime(macro_daily.index.date)
macro_daily

df_daily = pd.read_csv('all_gold_data.csv', index_col=0, parse_dates=True)
df_daily.sort_index(inplace=True)
df_daily = df_daily[~df_daily.index.duplicated(keep='first')]

# Normalize datetime index (remove time component)
df_daily.index = pd.to_datetime(df_daily.index.date)

df_daily['actual_close'] = df_daily['close'].copy()
df_daily['ln_close'] = np.log(df_daily['close'])

for col in ['open', 'high', 'low', 'close']:
    df_daily[col] = denoise_data(df_daily[col].values)

# Momentum & Trend
df_daily.ta.rsi(length=14, append=True)
df_daily.ta.macd(fast=12, slow=26, signal=9, append=True)
df_daily.ta.adx(length=14, append=True)
df_daily.ta.cci(length=20, append=True)

# Volatility & Bands
df_daily.ta.bbands(length=20, std=2, append=True)
df_daily.ta.atr(length=14, append=True)

# Moving Average Distances
df_daily.ta.ema(length=50, append=True)
df_daily.ta.ema(length=200, append=True)
df_daily['dist_ema50'] = (df_daily['close'] - df_daily['EMA_50']) / df_daily['EMA_50']
df_daily['dist_ema200'] = (df_daily['close'] - df_daily['EMA_200']) / df_daily['EMA_200']

# Statistical & Others
df_daily['daily_range'] = (df_daily['high'] - df_daily['low']) / df_daily['open']
rolling_mean = df_daily['close'].rolling(window=20).mean()
rolling_std = df_daily['close'].rolling(window=20).std()
df_daily['z_score'] = (df_daily['close'] - rolling_mean) / rolling_std

cols_to_drop = ['EMA_50', 'EMA_200', 'BBU_20_2.0', 'BBL_20_2.0', 'BBM_20_2.0']
df_daily.drop(columns=[c for c in cols_to_drop if c in df_daily.columns], inplace=True)
df_daily.dropna(inplace=True)
df_daily

"""
For joining daily-marco variable to technical data
"""

macro_to_merge = dollar_index.join(vix, how='inner')
macro_to_merge = macro_to_merge[macro_to_merge.index >= '2016-02-01']
macro_to_merge

df_final = macro_daily.join(df_daily, how='right')
df_final = df_final.join(macro_to_merge, how='left')
df_final = df_final.ffill()
df_final = df_final.dropna()
forecast_horizon = 5
for i in range(1, forecast_horizon + 1):
    col_name = f'target_return_{i}d'
    # df_final[col_name] = np.log(df_final['actual_close']).shift(-i) - np.log(df_final['close'])
    df_final[col_name] = np.log(df_final['close']).shift(-i) - np.log(df_final['close'])

df_predict_latest = df_final.tail(forecast_horizon).copy()

# threshold = 0.00
# choices = [1, 0]
# for i in range(1, forecast_horizon + 1):
#     target_col = f'target_return_{i}d'
#     signal_col = f'signal_{i}d'
    
#     conditions = [
#         (df_final[target_col] >= threshold),
#         (df_final[target_col] < -threshold)
#     ]
    
#     df_final[signal_col] = np.select(conditions, choices, default=0)


df_final.to_csv('gold_technical.csv', index=True)
df_final

df_final.columns

df = pd.read_csv('gold_technical.csv', index_col=0, parse_dates=True)

subset = df.tail(30) 

plt.figure(figsize=(12, 6))
plt.plot(subset.index, subset['actual_close'], 
         label='Actual Close (Raw Price)', 
         color='red', # Red
         linestyle='--', 
         marker='o', 
         markersize=4, 
         alpha=0.7)
plt.plot(subset.index, subset['close'], 
         label='Close (Denoised Trend)', 
         color='#F1C40F', # Gold
         linewidth=3, 
         marker='s', 
         markersize=4)
plt.title(f'Gold Price Comparison: Raw vs Denoised (Latest Period)', fontsize=14, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Price (USD)', fontsize=12)
plt.legend(loc='best')
plt.grid(True, linestyle=':', alpha=0.5, linewidth=1.2, color='gray')
plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
plt.gcf().autofmt_xdate() 
plt.tight_layout()
plt.show()

# --- ⚙️ Config ---
N_MODELS = 10
WINDOW_SIZE = 10      
BATCH_SIZE = 64       
MODEL_DIR = 'ensemble_huber_models' # 📁 ชื่อโฟลเดอร์ใหม่

os.makedirs(MODEL_DIR, exist_ok=True)

def set_seeds(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    tf.random.set_seed(seed)
    np.random.seed(seed)

print("🚀 Loading Data...")

# --- 1. Data Preparation ---
df = pd.read_csv('gold_technical.csv', index_col=0, parse_dates=True)
# df = df[:-5] 

df_train_full = df.dropna(subset=['target_return_5d']).copy()

# feature_cols = [c for c in df.columns if 'target' not in c and c != 'actual_close' and c != 'close' and c != 'macro_signal']
feature_cols = [c for c in df.columns if 'target' not in c and c != 'actual_close' and c != 'close']
target_cols = ['target_return_1d', 'target_return_2d', 'target_return_3d', 'target_return_4d', 'target_return_5d']

X = df_train_full[feature_cols].values
y = df_train_full[target_cols].values

train_size = int(len(X) * 0.8)
X_train_raw, X_test_raw = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

scaler_X = StandardScaler()
X_train_scaled = scaler_X.fit_transform(X_train_raw)
X_test_scaled = scaler_X.transform(X_test_raw)

def create_sequences(X, y, window_size):
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        Xs.append(X[i:(i + window_size)])
        ys.append(y[i + window_size])
    return np.array(Xs), np.array(ys)

X_train_seq, y_train_seq = create_sequences(X_train_scaled, y_train, WINDOW_SIZE)
X_test_seq, y_test_seq = create_sequences(X_test_scaled, y_test, WINDOW_SIZE)

# --- 2. Multi-Output Ensemble Training ---
print(f"\nTraining Huber Multi-Output Ensemble ({N_MODELS} Models)...")
model_files = []

for i in range(N_MODELS):
    print(f"   Training Model {i+1}/{N_MODELS}...")
    set_seeds(42 + i) # 🌟 สร้างความหลากหลายด้วย Seed ที่ต่างกัน
    
    model = Sequential([
        Conv1D(filters=32, kernel_size=1, padding='same', activation='swish', input_shape=(WINDOW_SIZE, len(feature_cols))),
        MaxPooling1D(pool_size=1),
        Bidirectional(LSTM(64, return_sequences=False, activation='tanh')),
        # LSTM(64, return_sequences=False, activation='tanh'),
        Dropout(0.3),
        Dense(64, activation='swish'),
        Dense(5) # 🎯 ทายรวดเดียว 5 วันเพื่อรักษาความต่อเนื่องของเทรนด์
    ])

    model.compile(optimizer=Adam(learning_rate=0.001), loss=Huber(), metrics=['mae'])
    # model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])
    # model.compile(optimizer=Adam(learning_rate=0.001), loss='mae', metrics=['mse'])
    
    
    filename = os.path.join(MODEL_DIR, f'gold_huber_ens_{i}.keras')
    model_files.append(filename)
    
    callbacks = [
        ModelCheckpoint(filename, save_best_only=True, monitor='val_loss', mode='min', verbose=0),
        EarlyStopping(monitor='val_loss', patience=25, restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6, verbose=0)
    ]

    model.fit(
        X_train_seq, y_train_seq,
        epochs=100, 
        batch_size=BATCH_SIZE,
        validation_data=(X_test_seq, y_test_seq),
        callbacks=callbacks,
        verbose=0
    )

print(f"\nAll ensemble models saved in '{MODEL_DIR}/'")

# --- 3. Prediction & CME Calendar ---
print(f"\nGenerating Forecast...")

last_window = df[feature_cols].tail(WINDOW_SIZE).values
last_window_scaled = scaler_X.transform(last_window).reshape(1, WINDOW_SIZE, len(feature_cols))

all_preds = []
for filename in model_files:
    m = tf.keras.models.load_model(filename)
    all_preds.append(m.predict(last_window_scaled, verbose=0)[0])

avg_returns = np.mean(all_preds, axis=0)
current_actual_price = df['actual_close'].iloc[-1]
last_date = df.index[-1]
predicted_prices = [current_actual_price * np.exp(ret) for ret in avg_returns]

# Market Calendar (CME Gold)
market_cal = mcal.get_calendar('CMEGlobex_Gold')
valid_days = market_cal.valid_days(start_date=last_date + pd.Timedelta(days=1), 
                                   end_date=last_date + pd.Timedelta(days=15))
next_5_days = valid_days[:5].tz_localize(None)

print("\n" + "="*55)
print(f"Latest Closing Price ({last_date.strftime('%Y-%m-%d')}): {current_actual_price:.2f}")
print(f"Final Ensemble Forecast (Huber 5-Output):")
print("-" * 55)
for i, (price, date) in enumerate(zip(predicted_prices, next_5_days), 1):
    trend = "🟢 UP" if price > current_actual_price else "🔴 DOWN"
    print(f"Day {i} ({date.strftime('%Y-%m-%d')}): {price:.2f}  {trend} ({price - current_actual_price:+.2f})")
print("="*55)

# --- 4. Final Accuracy Evaluation ---
print(f"\nCalculating Final Test RMSE...")
test_preds_all = []
for filename in model_files:
    m = tf.keras.models.load_model(filename)
    test_preds_all.append(m.predict(X_test_seq, verbose=0))

y_pred_avg = np.mean(test_preds_all, axis=0)
test_start_idx = train_size + WINDOW_SIZE
base_prices = df_train_full['close'].iloc[test_start_idx:].values[:len(y_pred_avg)].reshape(-1, 1)

print("\n" + "="*50)
print(f"Accuracy Summary (Final Ensemble)")
print("="*50)
total_rmse = []
for i in range(5):
    true_p = base_prices * np.exp(y_test_seq[:len(base_prices), i].reshape(-1, 1))
    pred_p = base_prices * np.exp(y_pred_avg[:len(base_prices), i].reshape(-1, 1))
    rmse = np.sqrt(mean_squared_error(true_p, pred_p))
    total_rmse.append(rmse)
    print(f"RMSE {i+1}d : ${rmse:.2f}")

print("-" * 50)
print(f"Combined RMSE (1-5d): ${np.mean(total_rmse):.2f}")
print("="*50)

# --- 1. Setup Test Data ---
# Locate the start of the test set in the original dataframe
test_start_idx = train_size + WINDOW_SIZE
test_df = df_train_full.iloc[test_start_idx:].copy()

# The actual closing price at time T (used as the base to calculate future prices from log returns)
base_prices = test_df['close'].values[:len(y_pred_avg)]

# --- 2. Calculate Historical Volatility (125-day Sigma) ---
# Per rules: Use the Standard Deviation of the closing price for the past 125 trading days
df_full_prices = df['actual_close']
sigma_125d_rolling = df_full_prices.rolling(window=125).std()

# Align sigma values with the test set timeframe
test_sigmas = sigma_125d_rolling.iloc[test_start_idx : test_start_idx + len(y_pred_avg)].values

# --- 3. Convert Returns to Prices & Calculate VAAE ---
vaae_results = []

for i in range(5): # Loop through Day 1 to Day 5 horizons
    # Realized Price (Actual) vs Predicted Price
    # Formula: Base Price * exp(return)
    actual_p = base_prices * np.exp(y_test_seq[:len(base_prices), i])
    pred_p = base_prices * np.exp(y_pred_avg[:len(base_prices), i])
    
    # Absolute Error |Predicted - Actual|
    abs_error = np.abs(pred_p - actual_p)
    
    # VAAE = Absolute Error / Sigma
    # This adjusts the error based on how "noisy" the market was at that time
    vaae_day = abs_error / test_sigmas
    vaae_results.append(vaae_day)

# --- 4. Consolidate into DataFrame ---
vaae_cols = [f'VAAE_{i+1}d' for i in range(5)]
vaae_df = pd.DataFrame(
    np.column_stack(vaae_results), 
    columns=vaae_cols, 
    index=test_df.index[:len(y_pred_avg)]
)

# Calculate the Average VAAE across all 5 days (Your Match 3 Score)
vaae_df['Average_VAAE_Match3'] = vaae_df.mean(axis=1)

# --- 5. Summary Report ---
print("="*65)
print("📊 HISTORICAL VAAE PERFORMANCE REPORT (TEST SET)")
print("="*65)
summary = vaae_df[vaae_cols].mean()
for col in vaae_cols:
    print(f"Mean {col:8} : {summary[col]:.4f}")

print("-" * 65)
overall_score = vaae_df['Average_VAAE_Match3'].mean()
print(f"🏆 OVERALL TEST SET VAAE SCORE: {overall_score:.4f}")
print("   (Lower is better | Goal: < 1.0)")
print("="*65)

# Display latest samples
print("\n🔍 Latest Test Set VAAE Samples:")
vaae_df

df_results = pd.DataFrame(index=test_df.index[:len(y_pred_avg)])
base_prices_flat = base_prices.flatten()

for i in range(5):
    actual_returns = np.exp(y_test_seq[:len(base_prices_flat), i])
    predict_returns = np.exp(y_pred_avg[:len(base_prices_flat), i])
    true_p = base_prices_flat * actual_returns
    pred_p = base_prices_flat * predict_returns
    
    df_results[f'Actual_{i+1}d'] = true_p
    df_results[f'Predict_{i+1}d'] = pred_p
df_results = df_results[df_results.index >= '2025-01-01']

cme = mcal.get_calendar('CMEGlobex_Gold')
all_trading_days = cme.valid_days(start_date=df_results.index.min(), 
                                  end_date=df_results.index.max() + pd.Timedelta(days=15))
all_trading_days = all_trading_days.tz_localize(None)

colors_actual = "#8E9295"
colors_pred = ['#F1C40F', '#E67E22', '#E74C3C', '#9B59B6', '#3498DB']

for i in range(1, 6):
    plt.figure(figsize=(12, 6))
    
    shifted_index = []
    for current_date in df_results.index:
        idx = np.searchsorted(all_trading_days, current_date)
        shifted_index.append(all_trading_days[idx + i])
    
    plt.plot(shifted_index, df_results[f'Actual_{i}d'], 
             label=f'Actual Price (T+{i})', color=colors_actual, linestyle='--', alpha=0.8)
    
    plt.plot(shifted_index, df_results[f'Predict_{i}d'], 
             label=f'Ensemble Predict (T+{i})', color=colors_pred[i-1], linewidth=2.5)
    
    plt.title(f'Comparison: Actual vs Predict - Horizon {i} Day(s) Ahead', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left')
    plt.grid(True, linestyle=':', alpha=0.5, linewidth=1.2, color='gray')
    plt.ylabel('Price (USD)')
    plt.xlabel('Market Realized Date')
    plt.savefig(f'actual_vs_predict_T_plus_{i}d.png', dpi=300)
    plt.tight_layout()
    plt.show()
