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
import warnings

warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(page_title="LTC Trading Bot", page_icon="🧠", layout="wide")

# --- НАСТРОЙКИ ---
TIME_WINDOW_MINUTES = 1.0
TIME_WINDOW_SEC = TIME_WINDOW_MINUTES * 60.0
NEURONS = 64
SEQ_LENGTH = 50
MAX_POINTS_ON_PLOT = 50
LEARNING_RATE = 0.0005
MODEL_PATH = f"ltc_trader_model_{TIME_WINDOW_MINUTES}m.pth"

# Симуляция торговли
PAPER_BALANCE = 1000.0 # Стартовый баланс
TRADE_AMOUNT = 1000.0  # Сумма каждой сделки

class RealtimeLTC(nn.Module):
    def __init__(self):
        super().__init__()
        wiring = AutoNCP(units=NEURONS, output_size=1)
        self.ltc = LTC(input_size=1, units=wiring, batch_first=True)

    def forward(self, x):
        out, _ = self.ltc(x)
        return out[:, -1, :]

# Инициализация стейта и фонового потока (только 1 раз)
@st.cache_resource
def start_bot_thread():
    state = {
        'price_history': [],
        'plot_real_prices': deque(maxlen=MAX_POINTS_ON_PLOT),
        'plot_pred_prices': deque(maxlen=MAX_POINTS_ON_PLOT),
        'plot_signals': deque(maxlen=MAX_POINTS_ON_PLOT),
        'plot_times': deque(maxlen=MAX_POINTS_ON_PLOT),
        'last_process_time': 0,
        'current_candle_prices': [],
        'ticks_counter': 0,
        'status': 'Запуск...',
        'latest_price': 0.0,
        'latest_pred': 0.0,
        'loss': 0.0,
        'progress': 0,
        # Данные симуляции
        'balance': PAPER_BALANCE,
        'position': 'NONE',
        'entry_price': 0.0,
        'unrealized_pnl': 0.0,
        'trades_count': 0
    }
    
    model = RealtimeLTC()
    if os.path.exists(MODEL_PATH):
        model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    model.train()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[50000], [100000]]))

    def fetch_historical_candles():
        interval = f"{int(TIME_WINDOW_MINUTES)}m"
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit={SEQ_LENGTH+1}"
        try:
            response = requests.get(url)
            if response.status_code == 200:
                data = response.json()
                for kline in data:
                    close_price = float(kline[4])
                    scaled_price = scaler.transform([[close_price]])[0][0]
                    state['price_history'].append([scaled_price])
                state['status'] = f"✅ История загружена ({len(state['price_history'])}/{SEQ_LENGTH+1})"
            else:
                state['status'] = "❌ Ошибка загрузки истории API"
        except Exception as e:
            state['status'] = f"❌ Ошибка: {e}"

    def execute_trade(action, price):
        # Если позиция уже открыта в этом направлении
        if state['position'] == action:
            return
            
        fee_rate = 0.001 # 0.1% комиссия Binance

        # Закрываем старую позицию
        if state['position'] != 'NONE':
            if state['position'] == 'LONG':
                profit_pct = (price - state['entry_price']) / state['entry_price']
            elif state['position'] == 'SHORT':
                profit_pct = (state['entry_price'] - price) / state['entry_price']
                
            state['balance'] += (TRADE_AMOUNT * profit_pct) - (TRADE_AMOUNT * fee_rate)

        # Открываем новую
        state['position'] = action
        state['entry_price'] = price
        state['balance'] -= (TRADE_AMOUNT * fee_rate)
        state['trades_count'] += 1

    def on_message(ws, message):
        try:
            data = json.loads(message)
            current_price = float(data['p'])
            current_time = time.time()
            
            state['latest_price'] = current_price
            state['current_candle_prices'].append(current_price)
            
            # Живой PnL (нереализованная прибыль открытой сделки)
            if state['position'] == 'LONG':
                state['unrealized_pnl'] = (current_price - state['entry_price']) / state['entry_price'] * TRADE_AMOUNT
            elif state['position'] == 'SHORT':
                state['unrealized_pnl'] = (state['entry_price'] - current_price) / state['entry_price'] * TRADE_AMOUNT
            else:
                state['unrealized_pnl'] = 0.0
            
            if state['last_process_time'] == 0:
                state['last_process_time'] = current_time
                
            elapsed_time = current_time - state['last_process_time']
            
            if elapsed_time >= TIME_WINDOW_SEC:
                state['last_process_time'] = current_time
                candle_close_price = state['current_candle_prices'][-1]
                state['current_candle_prices'].clear()
                
                scaled_price = scaler.transform([[candle_close_price]])[0][0]
                state['price_history'].append([scaled_price])
                
                if len(state['price_history']) > SEQ_LENGTH + 1:
                    state['price_history'].pop(0)
                    
                if len(state['price_history']) == SEQ_LENGTH + 1:
                    X_train = torch.tensor([state['price_history'][:-1]], dtype=torch.float32)
                    y_true = torch.tensor([[state['price_history'][-1]]], dtype=torch.float32)
                    
                    optimizer.zero_grad()
                    y_pred = model(X_train)
                    loss = criterion(y_pred, y_true)
                    loss.backward()
                    optimizer.step()
                    
                    state['loss'] = loss.item()
                    state['ticks_counter'] += 1
                    
                    if state['ticks_counter'] % 10 == 0:
                        torch.save(model.state_dict(), MODEL_PATH)
                    
                    X_test = torch.tensor([state['price_history'][1:]], dtype=torch.float32)
                    with torch.no_grad():
                        future_pred_scaled = model(X_test).numpy()
                    
                    pred_price = scaler.inverse_transform(future_pred_scaled)[0][0]
                    state['latest_pred'] = pred_price
                    
                    diff = pred_price - candle_close_price
                    signal = 'WAIT'
                    THRESHOLD = 50.0 # Увеличен порог: игнорируем мелкий шум, ждем сильных движений
                    
                    if diff > THRESHOLD: 
                        signal = 'BUY'
                        execute_trade('LONG', candle_close_price)
                    elif diff < -THRESHOLD: 
                        signal = 'SELL'
                        execute_trade('SHORT', candle_close_price)
                    
                    state['plot_real_prices'].append(candle_close_price)
                    state['plot_pred_prices'].append(pred_price)
                    state['plot_signals'].append(signal)
                    current_hh_mm_ss = time.strftime('%H:%M:%S')
                    state['plot_times'].append(current_hh_mm_ss)
                    
                    state['status'] = f"🟢 Свеча закрыта в {current_hh_mm_ss}."
            else:
                state['progress'] = int((elapsed_time / TIME_WINDOW_SEC) * 100)
                state['status'] = f"⏳ Сбор свечи... {state['progress']}%"
        except Exception:
            pass

    def on_error(ws, error):
        pass
    def on_close(ws, close_status_code, close_msg):
        pass
    def on_open(ws):
        fetch_historical_candles()

    def run_websocket():
        socket_url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
        while True:
            ws = websocket.WebSocketApp(socket_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
            ws.run_forever(ping_interval=60, ping_timeout=10)
            time.sleep(2)

    thread = threading.Thread(target=run_websocket, daemon=True)
    thread.start()
    return state

state = start_bot_thread()

st.title("🧠 Liquid Time-Constant (LTC) Trading Bot")
st.markdown(f"**Таймфрейм:** {TIME_WINDOW_MINUTES} минут | **Нейронов:** {NEURONS}")

# --- БАЛАНС И ТОРГОВЛЯ ---
st.subheader("💼 Симуляция торговли (Paper Trading)")
tc1, tc2, tc3, tc4 = st.columns(4)

total_equity = state['balance'] + state['unrealized_pnl']
profit_loss = total_equity - PAPER_BALANCE

tc1.metric("Доступный баланс", f"${state['balance']:.2f}")
tc2.metric("Нереализованный PnL", f"${state['unrealized_pnl']:.2f}")
tc3.metric("Общий капитал", f"${total_equity:.2f}", f"{profit_loss:+.2f}")

pos_color = "gray"
if state['position'] == 'LONG': pos_color = "green"
elif state['position'] == 'SHORT': pos_color = "red"

tc4.markdown(f"**Открытая позиция:** <span style='color:{pos_color}; font-weight:bold;'>{state['position']}</span> (Сделок: {state['trades_count']})", unsafe_allow_html=True)
if state['position'] != 'NONE':
    tc4.markdown(f"**Цена входа:** ${state['entry_price']:.2f}")


# --- МЕТРИКИ НЕЙРОСЕТИ ---
st.divider()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Текущая цена BTC", f"${state['latest_price']:.2f}")

diff = state['latest_pred'] - state['latest_price'] if state['latest_pred'] else 0
col2.metric("Прогноз LTC", f"${state['latest_pred']:.2f}", f"{diff:+.2f}")

col3.metric("Ошибка модели (Loss)", f"{state['loss']:.6f}" if state['loss'] else "0.000000")
col4.metric("Статус", state['status'])

st.progress(state['progress'] / 100)

# --- ГРАФИК ---
st.subheader("Живой график цены и прогнозов")

if len(state['plot_real_prices']) > 0:
    df = pd.DataFrame({
        'Время': list(state['plot_times']),
        'Реальная цена': list(state['plot_real_prices']),
        'Прогноз': list(state['plot_pred_prices'])
    }).set_index('Время')
    
    st.line_chart(df, color=["#2962FF", "#FF0000"])
    
    # Сигналы
    st.subheader("Последние торговые сигналы")
    has_signals = False
    for i in reversed(range(len(state['plot_signals']))):
        if state['plot_signals'][i] != 'WAIT':
            has_signals = True
            color = "🟢" if state['plot_signals'][i] == 'BUY' else "🔴"
            st.write(f"{color} **{state['plot_signals'][i]}** в {state['plot_times'][i]} при цене **${state['plot_real_prices'][i]:.2f}**")
    
    if not has_signals:
        st.write("Пока нет четких сигналов. Нейросеть считает, что движение слишком маленькое.")
else:
    st.info("Ожидание формирования свечей и начала предсказаний...")

# Автоматическое обновление интерфейса раз в 1 секунду
time.sleep(1)
st.rerun()