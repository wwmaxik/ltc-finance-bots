import streamlit as st
import json
import websocket
import torch
import torch.nn as nn
import numpy as np
from ncps.torch import LTC
from ncps.wirings import AutoNCP
from sklearn.preprocessing import MinMaxScaler
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
NUM_FEATURES = 5
NEURONS = 32
SEQ_LENGTH = 50
MODEL_PATH = f"ltc_pro_model_{TIME_WINDOW_MINUTES}m.pth"

class RealtimeLTC(nn.Module):
    def __init__(self, input_size=NUM_FEATURES):
        super().__init__()
        wiring = AutoNCP(units=NEURONS, output_size=1)
        self.ltc = LTC(input_size=input_size, units=wiring, batch_first=True)

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
        'max_points_on_plot': 100,
        'signals': deque(maxlen=500)
    }
    
    model = RealtimeLTC()
    if os.path.exists(MODEL_PATH):
        try:
            model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
            state['status'] = f"✅ Мозги загружены: {MODEL_PATH}"
        except Exception as e:
            state['status'] = f"⚠️ Не удалось загрузить модель: {e}"
            
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=state['learning_rate'])
    criterion = nn.MSELoss()
    
    scalers = {
        'close': MinMaxScaler(feature_range=(0, 1)),
        'volume': MinMaxScaler(feature_range=(0, 1)),
        'rsi': MinMaxScaler(feature_range=(0, 1)),
        'macd_hist': MinMaxScaler(feature_range=(0, 1)),
        'bbp': MinMaxScaler(feature_range=(0, 1)),
    }

    def fetch_historical_candles():
        interval = f"{int(TIME_WINDOW_MINUTES)}m"
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit=200"
        try:
            response = requests.get(url)
            df = pd.DataFrame(response.json(), columns=['time', 'open', 'high', 'low', 'close', 'volume', 'close_time', 'quote_av', 'trades', 'tb_base_av', 'tb_quote_av', 'ignore'])
            df = df[['time', 'open', 'high', 'low', 'close', 'volume']].astype(float)
            df['time'] = pd.to_datetime(df['time'], unit='ms')

            df.ta.rsi(length=14, append=True)
            df.ta.macd(fast=12, slow=26, signal=9, append=True)
            df.ta.bbands(length=20, append=True)
            df.rename(columns={'BBP_20_2.0': 'bbp', 'RSI_14': 'rsi', 'MACDh_12_26_9': 'macd_hist'}, inplace=True)
            df.dropna(inplace=True)

            for col in scalers:
                if col in df.columns:
                    scalers[col].fit(df[[col]])

            state['df_history'] = df.iloc[-100:].reset_index(drop=True)
            state['status'] = "✅ История и индикаторы загружены."
        except Exception as e:
            state['status'] = f"❌ Ошибка загрузки истории: {e}"

    def execute_trade(action, price):
        if state['position'] == action: return
        fee = state['trade_amount'] * 0.001
        
        if state['position'] != 'NONE':
            pnl = ((price - state['entry_price']) if state['position'] == 'LONG' else (state['entry_price'] - price)) / state['entry_price'] * state['trade_amount']
            state['balance'] += pnl - fee
            if pnl > 0: state['winning_trades'] += 1
            else: state['losing_trades'] += 1
        
        state['position'] = action
        state['entry_price'] = price
        state['balance'] -= fee
        state['trades_count'] += 1

    def on_message(ws, message):
        try:
            for pg in optimizer.param_groups: pg['lr'] = state['learning_rate']
            candle = json.loads(message)['k']

            if state['position'] != 'NONE':
                current_price = float(candle['c'])
                pnl = ((current_price - state['entry_price']) if state['position'] == 'LONG' else (state['entry_price'] - current_price)) / state['entry_price'] * state['trade_amount']
                state['unrealized_pnl'] = pnl
                if pnl <= -state['stop_loss_usd']:
                    execute_trade('NONE', current_price) # Close position
                    state['status'] = f"🚨 СРАБОТАЛ STOP-LOSS"

            if candle['x']:
                new_row = {'time': pd.to_datetime(candle['t'], unit='ms'), 'open': float(candle['o']), 'high': float(candle['h']), 'low': float(candle['l']), 'close': float(candle['c']), 'volume': float(candle['v'])}
                temp_df = pd.concat([state['df_history'], pd.DataFrame([new_row])], ignore_index=True)
                
                temp_df.ta.rsi(length=14, append=True, col_names=('rsi',))
                temp_df.ta.macd(fast=12, slow=26, signal=9, append=True, col_names=('macd', 'macd_hist', 'macd_signal'))
                temp_df.ta.bbands(length=20, append=True, col_names=('BBL', 'BBM', 'BBU', 'BBB', 'bbp'))
                temp_df.dropna(inplace=True)
                state['df_history'] = temp_df

                if len(temp_df) > SEQ_LENGTH:
                    df_slice = temp_df.tail(SEQ_LENGTH + 1)
                    features = np.hstack([scalers[col].transform(df_slice[[col]]) for col in scalers])
                    
                    X_train = torch.tensor([features[:-1]], dtype=torch.float32)
                    y_true = torch.tensor([[features[-1, 0]]], dtype=torch.float32)
                    
                    optimizer.zero_grad()
                    y_pred = model(X_train)
                    loss = criterion(y_pred, y_true)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    state['loss'] = loss.item()
                    
                    if state['ticks_counter'] % 10 == 0: torch.save(model.state_dict(), MODEL_PATH)
                    state['ticks_counter'] += 1

                    X_test = torch.tensor([features[1:]], dtype=torch.float32)
                    with torch.no_grad():
                        pred_scaled = model(X_test).numpy()
                    
                    pred_price = scalers['close'].inverse_transform(pred_scaled)[0][0]
                    state['latest_pred'] = pred_price
                    state['latest_price'] = new_row['close']
                    
                    diff = pred_price - new_row['close']
                    signal = 'WAIT'
                    if diff > state['threshold']: signal = 'BUY'
                    elif diff < -state['threshold']: signal = 'SELL'
                    
                    if signal != 'WAIT':
                        sma5 = temp_df['close'].tail(5).mean()
                        if (signal == 'BUY' and new_row['close'] < sma5) or (signal == 'SELL' and new_row['close'] > sma5):
                            signal = 'WAIT'
                        else:
                            execute_trade('LONG' if signal == 'BUY' else 'SHORT', new_row['close'])
                            state['signals'].append({'time': new_row['time'], 'type': signal, 'price': new_row['close']})

                    state['status'] = f"🟢 Свеча закрыта в {new_row['time'].strftime('%H:%M:%S')}."
        except Exception as e:
            state['status'] = f"Error: {e}"

    def run_websocket():
        socket_url = f"wss://stream.binance.com:9443/ws/btcusdt@kline_{int(TIME_WINDOW_MINUTES)}m"
        while True:
            ws = websocket.WebSocketApp(socket_url, on_open=lambda ws: fetch_historical_candles(), on_message=on_message)
            ws.run_forever(ping_interval=60, ping_timeout=10)
            time.sleep(2)

    thread = threading.Thread(target=run_websocket, daemon=True)
    thread.start()
    return state

state = start_bot_thread()

st.sidebar.title("⚙️ Панель управления")
state['learning_rate'] = st.sidebar.number_input("Скорость обучения (LR)", 0.0001, 0.1, state['learning_rate'], 0.0001, "%.4f")
state['threshold'] = st.sidebar.number_input("Порог сигнала (USD)", 1.0, 1000.0, state['threshold'], 5.0)
state['stop_loss_usd'] = st.sidebar.number_input("Stop-Loss (USD)", 1.0, 500.0, state['stop_loss_usd'], 1.0)
state['trade_amount'] = st.sidebar.number_input("Размер сделки (USD)", 10.0, 100000.0, state['trade_amount'], 100.0)
state['max_points_on_plot'] = st.sidebar.slider("Длина графика (свечей)", 10, 500, state['max_points_on_plot'], 10)

st.sidebar.divider()
uptime_str = str(datetime.timedelta(seconds=int(time.time() - state['start_time'])))
st.sidebar.metric("⏱ Аптайм", uptime_str)

st.title("🤖 LTC Pro Trading Bot")
st.markdown(f"**Таймфрейм:** {TIME_WINDOW_MINUTES} мин | **Нейронов:** {NEURONS} | **Фичей:** {NUM_FEATURES}")

tc1, tc2, tc3, tc4 = st.columns(4)
total_equity = state['balance'] + state['unrealized_pnl']
tc1.metric("Доступный баланс", f"${state['balance']:.2f}")
tc2.metric("Нереализованный PnL", f"${state['unrealized_pnl']:.2f}")
tc3.metric("Общий капитал", f"${total_equity:.2f}", f"{total_equity - 1000.0:+.2f}")
pos_color = "green" if state['position'] == 'LONG' else "red" if state['position'] == 'SHORT' else "gray"
tc4.markdown(f"**Позиция:** <span style='color:{pos_color};'>{state['position']}</span> @ ${state['entry_price']:.2f}", True)

sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Всего сделок", state['trades_count'])
sc2.metric("Успешных", state['winning_trades'])
sc3.metric("Убыточных", state['losing_trades'])
win_rate = (state['winning_trades'] / state['trades_count'] * 100) if state['trades_count'] > 0 else 0
sc4.metric("Win Rate", f"{win_rate:.1f}%")

st.divider()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Текущая цена BTC", f"${state['latest_price']:.2f}")
col2.metric("Прогноз LTC", f"${state['latest_pred']:.2f}", f"{state['latest_pred'] - state['latest_price']:+.2f}")
col3.metric("Ошибка (Loss)", f"{state['loss']:.6f}")
col4.metric("Статус", state['status'])

df_plot = state['df_history'].tail(state['max_points_on_plot'])
if not df_plot.empty:
    df_plot['prediction'] = np.nan
    if state['latest_pred'] > 0:
        df_plot.loc[df_plot.index[-1], 'prediction'] = state['latest_pred']
    
    st.subheader("Живой график цены и прогнозов")
    st.line_chart(df_plot[['close', 'prediction']], color=["#2962FF", "#FF0000"])

    st.subheader("Индикаторы")
    c1, c2 = st.columns(2)
    c1.line_chart(df_plot['rsi'])
    c2.bar_chart(df_plot['macd_hist'])
    
    st.subheader("История сигналов")
    if state['signals']:
        for s in reversed(state['signals']):
            st.write(f"{'🟢' if s['type'] == 'BUY' else '🔴'} **{s['type']}** в {s['time'].strftime('%H:%M:%S')} при цене ${s['price']:.2f}")
    else:
        st.info("Пока не было сигналов.")
else:
    st.info("Ожидание данных от Binance...")

time.sleep(1)
st.rerun()
