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
import datetime

warnings.filterwarnings('ignore')

# Настройка страницы
st.set_page_config(page_title="LTC Trading Bot", page_icon="🧠", layout="wide")

# --- КОНСТАНТЫ ИНИЦИАЛИЗАЦИИ ---
TIME_WINDOW_MINUTES = 1.0
TIME_WINDOW_SEC = TIME_WINDOW_MINUTES * 60.0
NEURONS = 32
SEQ_LENGTH = 50
MODEL_PATH = f"ltc_trader_model_{TIME_WINDOW_MINUTES}m_v2.pth"

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
        'plot_real_prices': deque(maxlen=1000), # Максимум 1000 точек в памяти для длинного графика
        'plot_pred_prices': deque(maxlen=1000),
        'plot_signals': deque(maxlen=1000),
        'plot_times': deque(maxlen=1000),
        'last_process_time': 0,
        'current_candle_prices': [],
        'ticks_counter': 0,
        'status': 'Запуск...',
        'latest_price': 0.0,
        'latest_pred': 0.0,
        'loss': 0.0,
        'progress': 0,
        
        # Симуляция
        'balance': 1000.0,
        'position': 'NONE',
        'entry_price': 0.0,
        'unrealized_pnl': 0.0,
        
        # Статистика
        'trades_count': 0,
        'winning_trades': 0,
        'losing_trades': 0,
        'start_time': time.time(),
        
        # Динамические настройки (могут меняться из UI)
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
        if state['position'] == action:
            return
            
        fee_rate = 0.001

        if state['position'] != 'NONE':
            if state['position'] == 'LONG':
                profit_pct = (price - state['entry_price']) / state['entry_price']
            elif state['position'] == 'SHORT':
                profit_pct = (state['entry_price'] - price) / state['entry_price']
                
            pnl = (state['trade_amount'] * profit_pct) - (state['trade_amount'] * fee_rate)
            state['balance'] += pnl
            
            if pnl > 0: state['winning_trades'] += 1
            else: state['losing_trades'] += 1

        state['position'] = action
        state['entry_price'] = price
        state['balance'] -= (state['trade_amount'] * fee_rate)
        state['trades_count'] += 1

    def on_message(ws, message):
        try:
            # Обновляем LR оптимизатора на лету, если ползунок в UI сдвинули
            for param_group in optimizer.param_groups:
                param_group['lr'] = state['learning_rate']
                
            data = json.loads(message)
            current_price = float(data['p'])
            current_time = time.time()
            
            state['latest_price'] = current_price
            state['current_candle_prices'].append(current_price)
            
            if state['position'] == 'LONG':
                state['unrealized_pnl'] = (current_price - state['entry_price']) / state['entry_price'] * state['trade_amount']
            elif state['position'] == 'SHORT':
                state['unrealized_pnl'] = (state['entry_price'] - current_price) / state['entry_price'] * state['trade_amount']
            else:
                state['unrealized_pnl'] = 0.0
                
            if state['position'] != 'NONE' and state['unrealized_pnl'] <= -state['stop_loss_usd']:
                state['balance'] += state['unrealized_pnl'] - (state['trade_amount'] * 0.001)
                state['losing_trades'] += 1
                state['position'] = 'NONE'
                state['entry_price'] = 0.0
                state['unrealized_pnl'] = 0.0
                state['status'] = f"🚨 СРАБОТАЛ STOP-LOSS при цене {current_price:.2f}$"
            
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
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
                    
                    if diff > state['threshold']: 
                        signal = 'BUY'
                        if len(state['plot_real_prices']) > 5:
                            sma5 = sum(list(state['plot_real_prices'])[-5:]) / 5
                            if candle_close_price < sma5: signal = 'WAIT'
                        if signal == 'BUY': execute_trade('LONG', candle_close_price)
                        
                    elif diff < -state['threshold']: 
                        signal = 'SELL'
                        if len(state['plot_real_prices']) > 5:
                            sma5 = sum(list(state['plot_real_prices'])[-5:]) / 5
                            if candle_close_price > sma5: signal = 'WAIT'
                        if signal == 'SELL': execute_trade('SHORT', candle_close_price)
                    
                    state['plot_real_prices'].append(candle_close_price)
                    state['plot_pred_prices'].append(pred_price)
                    state['plot_signals'].append(signal)
                    current_hh_mm_ss = time.strftime('%H:%M:%S')
                    state['plot_times'].append(current_hh_mm_ss)
                    
                    if not state['status'].startswith("🚨"):
                        state['status'] = f"🟢 Свеча закрыта в {current_hh_mm_ss}."
            else:
                state['progress'] = int((elapsed_time / TIME_WINDOW_SEC) * 100)
                if not state['status'].startswith("🚨"):
                    state['status'] = f"⏳ Сбор свечи... {state['progress']}%"
        except Exception as e:
            pass

    def on_error(ws, error): pass
    def on_close(ws, close_status_code, close_msg): pass
    def on_open(ws): fetch_historical_candles()

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

# --- САЙДБАР (Управление нейронкой) ---
st.sidebar.title("⚙️ Панель управления")
st.sidebar.markdown("Настройки применяются к нейросети **мгновенно**.")

state['learning_rate'] = st.sidebar.number_input("Скорость обучения (LR)", min_value=0.0001, max_value=0.1, value=state['learning_rate'], step=0.0001, format="%.4f")
state['threshold'] = st.sidebar.number_input("Порог сигнала (Threshold USD)", min_value=1.0, max_value=1000.0, value=state['threshold'], step=5.0)
state['stop_loss_usd'] = st.sidebar.number_input("Stop-Loss (Убыток в USD)", min_value=1.0, max_value=500.0, value=state['stop_loss_usd'], step=1.0)
state['trade_amount'] = st.sidebar.number_input("Размер сделки (USD)", min_value=10.0, max_value=100000.0, value=state['trade_amount'], step=100.0)
state['max_points_on_plot'] = st.sidebar.slider("Длина графика (свечей)", min_value=10, max_value=1000, value=state['max_points_on_plot'], step=10)

st.sidebar.divider()

# Аптайм
uptime_sec = int(time.time() - state['start_time'])
uptime_str = str(datetime.timedelta(seconds=uptime_sec))
st.sidebar.metric("⏱ Аптайм бота", uptime_str)

st.title("🧠 Liquid Time-Constant (LTC) Trading Bot")
st.markdown(f"**Таймфрейм:** {TIME_WINDOW_MINUTES} минут | **Нейронов:** {NEURONS}")

# --- БАЛАНС И ТОРГОВЛЯ ---
st.subheader("💼 Симуляция торговли (Paper Trading) и Статистика")
tc1, tc2, tc3, tc4 = st.columns(4)

total_equity = state['balance'] + state['unrealized_pnl']
profit_loss = total_equity - 1000.0

tc1.metric("Доступный баланс", f"${state['balance']:.2f}")
tc2.metric("Нереализованный PnL", f"${state['unrealized_pnl']:.2f}")
tc3.metric("Общий капитал", f"${total_equity:.2f}", f"{profit_loss:+.2f}")

pos_color = "gray"
if state['position'] == 'LONG': pos_color = "green"
elif state['position'] == 'SHORT': pos_color = "red"

tc4.markdown(f"**Открытая позиция:** <span style='color:{pos_color}; font-weight:bold;'>{state['position']}</span>", unsafe_allow_html=True)
if state['position'] != 'NONE':
    tc4.markdown(f"**Цена входа:** ${state['entry_price']:.2f}")

# --- ПОЛНАЯ СТАТИСТИКА ---
sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Всего сделок", state['trades_count'])
sc2.metric("Успешных сделок (Win)", state['winning_trades'])
sc3.metric("Убыточных сделок (Loss)", state['losing_trades'])
win_rate = (state['winning_trades'] / state['trades_count'] * 100) if state['trades_count'] > 0 else 0.0
sc4.metric("Win Rate", f"{win_rate:.1f}%")

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

# Берем нужное количество точек для графика в зависимости от настройки ползунка
points_to_show = min(len(state['plot_real_prices']), state['max_points_on_plot'])

if points_to_show > 0:
    df = pd.DataFrame({
        'Время': list(state['plot_times'])[-points_to_show:],
        'Реальная цена': list(state['plot_real_prices'])[-points_to_show:],
        'Прогноз': list(state['plot_pred_prices'])[-points_to_show:]
    }).set_index('Время')
    
    st.line_chart(df, color=["#2962FF", "#FF0000"])
    
    # Сигналы
    st.subheader("Последние торговые сигналы")
    has_signals = False
    
    signals_list = list(state['plot_signals'])[-points_to_show:]
    times_list = list(state['plot_times'])[-points_to_show:]
    real_prices_list = list(state['plot_real_prices'])[-points_to_show:]
    
    for i in reversed(range(len(signals_list))):
        if signals_list[i] != 'WAIT':
            has_signals = True
            color = "🟢" if signals_list[i] == 'BUY' else "🔴"
            st.write(f"{color} **{signals_list[i]}** в {times_list[i]} при цене **${real_prices_list[i]:.2f}**")
    
    if not has_signals:
        st.write("На выбранном отрезке графика нет сигналов. Нейросеть считает, что движение слишком маленькое.")
else:
    st.info("Ожидание формирования свечей и начала предсказаний...")

# Автоматическое обновление интерфейса раз в 1 секунду
time.sleep(1)
st.rerun()