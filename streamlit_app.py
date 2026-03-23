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

st.set_page_config(page_title="LTC Pro Bot", page_icon="🤖", layout="wide")

TIME_WINDOW_MINUTES = 1.0
TIME_WINDOW_SEC = TIME_WINDOW_MINUTES * 60.0
NUM_FEATURES = 2
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
        'price_history': [],
        'volume_history': [],
        'plot_real_prices': deque(maxlen=1000),
        'plot_pred_prices': deque(maxlen=1000),
        'plot_signals': deque(maxlen=1000),
        'plot_times': deque(maxlen=1000),
        'current_candle_prices': [],
        'current_candle_volume': 0.0,
        'last_process_time': 0,
        'ticks_counter': 0,
        'status': 'Запуск...',
        'latest_price': 0.0,
        'latest_volume': 0.0,
        'latest_pred': 0.0,
        'loss': 0.0,
        'progress': 0,
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
    
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(np.array([[60000, 0], [80000, 500]]))

    def fetch_historical_candles():
        interval = f"{int(TIME_WINDOW_MINUTES)}m"
        url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit={SEQ_LENGTH+1}"
        try:
            response = requests.get(url, timeout=15)
            if response.status_code == 200:
                data = response.json()
                for kline in data:
                    close_price = float(kline[4])
                    volume = float(kline[5])
                    scaled = scaler.transform([[close_price, volume]])[0]
                    state['price_history'].append([scaled[0], scaled[1]])
                    state['volume_history'].append(volume)
                    state['plot_real_prices'].append(close_price)
                    state['plot_pred_prices'].append(0)
                    state['plot_signals'].append('WAIT')
                    state['plot_times'].append(datetime.datetime.fromtimestamp(kline[0]/1000).strftime('%H:%M:%S'))
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
            else:
                profit_pct = 0
                
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
            for pg in optimizer.param_groups: pg['lr'] = state['learning_rate']
            data = json.loads(message)
            current_price = float(data['p'])
            current_quantity = float(data['q'])
            current_time = time.time()
            
            state['latest_price'] = current_price
            state['current_candle_prices'].append(current_price)
            state['current_candle_volume'] += current_quantity
            
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
                candle_volume = state['current_candle_volume']
                state['current_candle_prices'].clear()
                state['current_candle_volume'] = 0.0
                
                scaled = scaler.transform([[candle_close_price, candle_volume]])[0]
                state['price_history'].append([scaled[0], scaled[1]])
                state['volume_history'].append(candle_volume)
                
                if len(state['price_history']) > SEQ_LENGTH + 1:
                    state['price_history'].pop(0)
                    state['volume_history'].pop(0)
                    
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
                    
                    pred_price = scaler.inverse_transform([[future_pred_scaled[0][0], 0]])[0][0]
                    state['latest_pred'] = pred_price
                    state['latest_volume'] = candle_volume
                    
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
                    
                    if signal != 'WAIT':
                        state['signals'].append({'time': current_hh_mm_ss, 'type': signal, 'price': candle_close_price})
                    
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

st.sidebar.title("⚙️ Управление")
state['learning_rate'] = st.sidebar.number_input("LR", 0.0001, 0.1, state['learning_rate'], 0.0001, "%.4f")
state['threshold'] = st.sidebar.number_input("Порог (USD)", 1.0, 1000.0, state['threshold'], 5.0)
state['stop_loss_usd'] = st.sidebar.number_input("Stop-Loss", 1.0, 500.0, state['stop_loss_usd'], 1.0)
state['trade_amount'] = st.sidebar.number_input("Сделка (USD)", 10.0, 100000.0, state['trade_amount'], 100.0)
state['max_points_on_plot'] = st.sidebar.slider("Свечей на графике", 10, 500, state['max_points_on_plot'], 10)

st.sidebar.divider()
uptime_str = str(datetime.timedelta(seconds=int(time.time() - state['start_time'])))
st.sidebar.metric("⏱ Аптайм", uptime_str)

st.title("🤖 LTC Pro Trading Bot")
st.markdown(f"**{TIME_WINDOW_MINUTES}м** | Нейроны: {NEURONS} | Фичи: price + volume")

tc1, tc2, tc3, tc4 = st.columns(4)
total_equity = state['balance'] + state['unrealized_pnl']
tc1.metric("Баланс", f"${state['balance']:.2f}")
tc2.metric("PnL", f"${state['unrealized_pnl']:.2f}")
tc3.metric("Капитал", f"${total_equity:.2f}", f"{total_equity - 1000.0:+.2f}")
pos_color = "green" if state['position'] == 'LONG' else "red" if state['position'] == 'SHORT' else "gray"
tc4.markdown(f"**Позиция:** <span style='color:{pos_color}'>{state['position']}</span> @ ${state['entry_price']:.0f}", True)

sc1, sc2, sc3, sc4 = st.columns(4)
sc1.metric("Сделок", state['trades_count'])
sc2.metric("Win", state['winning_trades'])
sc3.metric("Loss", state['losing_trades'])
win_rate = (state['winning_trades'] / state['trades_count'] * 100) if state['trades_count'] > 0 else 0
sc4.metric("Win Rate", f"{win_rate:.1f}%")

st.divider()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Цена BTC", f"${state['latest_price']:.2f}")
col2.metric("Объём свечи", f"{state['latest_volume']:.4f} BTC")
col3.metric("Прогноз", f"${state['latest_pred']:.2f}", f"{state['latest_pred'] - state['latest_price']:+.2f}")
col4.metric("Loss", f"{state['loss']:.6f}")

st.caption(f"**Статус:** {state['status']}")
st.progress(state['progress'] / 100, text=f"⏳ Сбор свечи... {state['progress']}%")

points_to_show = min(len(state['plot_real_prices']), state['max_points_on_plot'])
if points_to_show > 0:
    df = pd.DataFrame({
        'Время': list(state['plot_times'])[-points_to_show:],
        'Цена': list(state['plot_real_prices'])[-points_to_show:],
        'Прогноз': list(state['plot_pred_prices'])[-points_to_show:]
    }).set_index('Время')
    
    st.subheader("График")
    st.line_chart(df, color=["#2962FF", "#FF0000"])
    
    st.subheader("Сигналы")
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
        st.write("Нет сигналов на выбранном отрезке.")
else:
    st.info("Ожидание формирования свечей...")

time.sleep(1)
st.rerun()
