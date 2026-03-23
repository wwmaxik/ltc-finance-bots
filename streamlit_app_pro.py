import streamlit as st
import json
import websocket
import torch
import torch.nn as nn
import numpy as np
from ncps.torch import LTC
from ncps.wirings import AutoNCP
from sklearn.preprocessing import MinMaxScaler
from collections import deque
import time
import os
import requests
import threading
import pandas as pd
import pandas_ta as ta
import warnings
import datetime

warnings.filterwarnings('ignore')

st.set_page_config(page_title="LTC Pro Bot", page_icon="🤖", layout="wide")

# --- НАСТРОЙКИ ---
TIME_WINDOW_MINUTES = 1.0
NUM_FEATURES = 5 # Теперь у нас 5 фичей: Close, Volume, RSI, MACD_hist, BB_percent
NEURONS = 32
SEQ_LENGTH = 50
MODEL_PATH = f"ltc_pro_model_{TIME_WINDOW_MINUTES}m.pth"

class RealtimeLTC(nn.Module):
    def __init__(self):
        super().__init__()
        wiring = AutoNCP(units=NEURONS, output_size=1) # Предсказываем все равно 1 значение - будущую цену
        self.ltc = LTC(input_size=NUM_FEATURES, units=wiring, batch_first=True)

    def forward(self, x):
        out, _ = self.ltc(x)
        return out[:, -1, :]

@st.cache_resource
def start_bot_thread():
    state = {
        'df_history': pd.DataFrame(),
        'status': 'Запуск...',
        'latest_price': 0.0,
        'latest_pred': 0.0,
        'loss': 0.0,
        'balance': 1000.0,
        'position': 'NONE',
        'entry_price': 0.0,
        'unrealized_pnl': 0.0,
        'trades_count': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'start_time': time.time(),
        'learning_rate': 0.001,
        'threshold': 150.0,
        'stop_loss_usd': 15.0,
        'trade_amount': 1000.0,
        'max_points_on_plot': 100
    }
    
    model = RealtimeLTC()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=state['learning_rate'])
    criterion = nn.MSELoss()
    
    # Теперь у нас несколько скейлеров, для каждой фичи свой
    scalers = {
        'close': MinMaxScaler(feature_range=(0, 1)),
        'volume': MinMaxScaler(feature_range=(0, 1)),
        'rsi': MinMaxScaler(feature_range=(0, 1)),
        'macd_hist': MinMaxScaler(feature_range=(0, 1)),
        'bbp': MinMaxScaler(feature_range=(0, 1)),
    }

    def fetch_historical_candles():
        interval = f"{int(TIME_WINDOW_MINUTES)}m"
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit=200" # Загрузим больше истории для индикаторов
        try:
            response = requests.get(url)
            data = response.json()
            
            temp_df = pd.DataFrame(data, columns=['kline_open_time', 'open', 'high', 'low', 'close', 'volume', 'kline_close_time', 'quote_asset_volume', 'number_of_trades', 'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'])
            temp_df = temp_df[['open', 'high', 'low', 'close', 'volume']].astype(float)
            
            # Рассчитываем индикаторы
            temp_df.ta.rsi(length=14, append=True)
            temp_df.ta.macd(fast=12, slow=26, signal=9, append=True)
            temp_df.ta.bbands(length=20, append=True)
            temp_df.rename(columns={'BBP_20_2.0': 'bbp', 'RSI_14': 'rsi', 'MACDh_12_26_9': 'macd_hist'}, inplace=True)
            
            state['df_history'] = temp_df.iloc[-100:].dropna().reset_index(drop=True)
            
            # Обучаем скейлеры на исторических данных
            for col in scalers:
                if col in state['df_history'].columns:
                    scalers[col].fit(state['df_history'][[col]])

            state['status'] = f"✅ История и индикаторы загружены."
        except Exception as e:
            state['status'] = f"❌ Ошибка загрузки истории: {e}"

    def execute_trade(action, price):
        # ... (логика торговли остается той же)
        pass

    def on_message(ws, message):
        try:
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']
            
            candle = json.loads(message)['k']
            if candle['x']: # Свеча закрылась
                new_row = {'open': float(candle['o']), 'high': float(candle['h']), 'low': float(candle['l']), 'close': float(candle['c']), 'volume': float(candle['v'])}
                state['df_history'].loc[len(state['df_history'])] = new_row
                
                # Пересчитываем индикаторы
                state['df_history'].ta.rsi(length=14, append=True)
                state['df_history'].ta.macd(fast=12, slow=26, signal=9, append=True)
                state['df_history'].ta.bbands(length=20, append=True)
                state['df_history'].rename(columns={'BBP_20_2.0': 'bbp', 'RSI_14': 'rsi', 'MACDh_12_26_9': 'macd_hist'}, inplace=True)
                state['df_history'].dropna(inplace=True)
                
                if len(state['df_history']) > SEQ_LENGTH:
                    # Подготовка данных для нейросети
                    df_slice = state['df_history'].tail(SEQ_LENGTH + 1)
                    
                    features = []
                    for col in scalers:
                        scaled_col = scalers[col].transform(df_slice[[col]])
                        features.append(scaled_col)
                    
                    scaled_features = np.hstack(features)
                    
                    # Обучение
                    X_train = torch.tensor([scaled_features[:-1]], dtype=torch.float32)
                    y_true = torch.tensor([[scaled_features[-1, 0]]], dtype=torch.float32) # Предсказываем только close
                    
                    optimizer.zero_grad()
                    y_pred = model(X_train)
                    loss = criterion(y_pred, y_true)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    state['loss'] = loss.item()
                    
                    # Предсказание
                    X_test = torch.tensor([scaled_features[1:]], dtype=torch.float32)
                    with torch.no_grad():
                        future_pred_scaled = model(X_test).numpy()
                    
                    pred_price = scalers['close'].inverse_transform(future_pred_scaled)[0][0]
                    state['latest_pred'] = pred_price
                    state['latest_price'] = new_row['close']
                    state['status'] = f"🟢 Свеча закрыта в {datetime.datetime.now().strftime('%H:%M:%S')}."
                    
        except Exception as e:
            state['status'] = f"Error: {e}"
            pass

    def run_websocket():
        # Переключаемся на стрим свечей
        socket_url = f"wss://stream.binance.com:9443/ws/btcusdt@kline_{int(TIME_WINDOW_MINUTES)}m"
        while True:
            ws = websocket.WebSocketApp(socket_url, on_open=lambda ws: fetch_historical_candles(), on_message=on_message)
            ws.run_forever(ping_interval=60, ping_timeout=10)
            time.sleep(2)

    thread = threading.Thread(target=run_websocket, daemon=True)
    thread.start()
    return state

state = start_bot_thread()

# --- UI ---
st.sidebar.title("⚙️ Панель управления")
state['learning_rate'] = st.sidebar.number_input("Скорость обучения (LR)", min_value=0.0001, max_value=0.1, value=state['learning_rate'], step=0.0001, format="%.4f")
# ... (остальные UI элементы) ...

st.title("🤖 LTC Pro Trading Bot")
# ... (отображение метрик) ...

# Графики
st.subheader("Живой график цены и прогнозов")
if not state['df_history'].empty:
    df_plot = state['df_history'].tail(state['max_points_on_plot']).copy()
    df_plot['prediction'] = np.nan # Создаем колонку для прогноза
    if state['latest_pred'] > 0:
        df_plot.loc[df_plot.index[-1], 'prediction'] = state['latest_pred']

    st.line_chart(df_plot[['close', 'prediction']], color=["#2962FF", "#FF0000"])

    st.subheader("Индикаторы")
    c1, c2 = st.columns(2)
    c1.line_chart(df_plot['rsi'])
    c2.bar_chart(df_plot['macd_hist'])

time.sleep(1)
st.rerun()