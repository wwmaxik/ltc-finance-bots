import streamlit as st
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
import pandas as pd
import warnings
import datetime

warnings.filterwarnings('ignore')

st.set_page_config(page_title="LTC Pro Bot", page_icon="🤖", layout="wide")

TIME_WINDOW_MINUTES = 1.0
NUM_FEATURES = 2
NEURONS = 32
SEQ_LENGTH = 50
MODEL_PATH = f"ltc_pro_model_{TIME_WINDOW_MINUTES}m.pth"
POLL_INTERVAL = 5

class RealtimeLTC(nn.Module):
    def __init__(self, input_size=NUM_FEATURES):
        super().__init__()
        wiring = AutoNCP(units=NEURONS, output_size=1)
        self.ltc = LTC(input_size=input_size, units=wiring, batch_first=True)

    def forward(self, x):
        out, _ = self.ltc(x)
        return out[:, -1, :]

def init_session_state():
    if 'initialized' not in st.session_state:
        st.session_state.initialized = True
        st.session_state.model = RealtimeLTC()
        if os.path.exists(MODEL_PATH):
            try:
                st.session_state.model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
            except:
                pass
        st.session_state.model.train()
        st.session_state.optimizer = torch.optim.Adam(st.session_state.model.parameters(), lr=0.001)
        st.session_state.criterion = nn.MSELoss()
        st.session_state.scaler = MinMaxScaler(feature_range=(0, 1))
        st.session_state.scaler.fit(np.array([[60000, 0], [80000, 500]]))
        st.session_state.price_history = []
        st.session_state.volume_history = []
        st.session_state.plot_real_prices = deque(maxlen=500)
        st.session_state.plot_pred_prices = deque(maxlen=500)
        st.session_state.plot_signals = deque(maxlen=500)
        st.session_state.plot_times = deque(maxlen=500)
        st.session_state.ticks_counter = 0
        st.session_state.last_candle_time = 0
        st.session_state.signals = []
        st.session_state.status = 'Загрузка...'
        st.session_state.learning_rate = 0.001
        st.session_state.threshold = 150.0
        st.session_state.stop_loss_usd = 15.0
        st.session_state.trade_amount = 1000.0
        st.session_state.max_points_on_plot = 100
        st.session_state.balance = 1000.0
        st.session_state.position = 'NONE'
        st.session_state.entry_price = 0.0
        st.session_state.unrealized_pnl = 0.0
        st.session_state.trades_count = 0
        st.session_state.winning_trades = 0
        st.session_state.losing_trades = 0
        st.session_state.start_time = time.time()
        st.session_state.latest_price = 0.0
        st.session_state.latest_volume = 0.0
        st.session_state.latest_pred = 0.0
        st.session_state.loss = 0.0
        load_initial_data()

def load_initial_data():
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={int(TIME_WINDOW_MINUTES)}m&limit={SEQ_LENGTH+1}"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, timeout=15, headers=headers)
        st.session_state.api_status = f"API: {response.status_code}"
        if response.status_code == 200:
            data = response.json()
            for kline in data:
                close_price = float(kline[4])
                volume = float(kline[5])
                scaled = st.session_state.scaler.transform([[close_price, volume]])[0]
                st.session_state.price_history.append([scaled[0], scaled[1]])
                st.session_state.volume_history.append(volume)
                st.session_state.plot_real_prices.append(close_price)
                st.session_state.plot_pred_prices.append(0)
                st.session_state.plot_signals.append('WAIT')
                st.session_state.plot_times.append(datetime.datetime.fromtimestamp(kline[0]/1000).strftime('%H:%M'))
            st.session_state.last_candle_time = data[-1][0]
            st.session_state.status = f"✅ Загружено {len(st.session_state.price_history)} свечей"
        else:
            st.session_state.status = f"❌ HTTP {response.status_code}"
    except Exception as e:
        st.session_state.status = f"❌ {type(e).__name__}"
        st.session_state.api_error = str(e)[:100]

def poll_new_data():
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={int(TIME_WINDOW_MINUTES)}m&limit=2"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, timeout=10, headers=headers)
        if response.status_code == 200:
            data = response.json()
            last_candle_time = int(data[-1][0])
            
            if last_candle_time > st.session_state.last_candle_time:
                st.session_state.last_candle_time = last_candle_time
                kline = data[-1]
                close_price = float(kline[4])
                volume = float(kline[5])
                
                scaled = st.session_state.scaler.transform([[close_price, volume]])[0]
                st.session_state.price_history.append([scaled[0], scaled[1]])
                st.session_state.volume_history.append(volume)
                
                if len(st.session_state.price_history) > SEQ_LENGTH + 1:
                    st.session_state.price_history.pop(0)
                    st.session_state.volume_history.pop(0)
                
                if len(st.session_state.price_history) == SEQ_LENGTH + 1:
                    train_model(close_price, volume, kline)
    except Exception as e:
        st.session_state.status = f"⚠️ {e}"

def train_model(close_price, volume, kline):
    for pg in st.session_state.optimizer.param_groups:
        pg['lr'] = st.session_state.learning_rate
    
    X_train = torch.tensor([st.session_state.price_history[:-1]], dtype=torch.float32)
    y_true = torch.tensor([[st.session_state.price_history[-1]]], dtype=torch.float32)
    
    st.session_state.optimizer.zero_grad()
    y_pred = st.session_state.model(X_train)
    loss = st.session_state.criterion(y_pred, y_true)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(st.session_state.model.parameters(), max_norm=1.0)
    st.session_state.optimizer.step()
    
    st.session_state.loss = loss.item()
    st.session_state.ticks_counter += 1
    
    if st.session_state.ticks_counter % 10 == 0:
        torch.save(st.session_state.model.state_dict(), MODEL_PATH)
    
    X_test = torch.tensor([st.session_state.price_history[1:]], dtype=torch.float32)
    with torch.no_grad():
        future_pred_scaled = st.session_state.model(X_test).numpy()
    
    pred_price = st.session_state.scaler.inverse_transform([[future_pred_scaled[0][0], 0]])[0][0]
    st.session_state.latest_pred = pred_price
    st.session_state.latest_price = close_price
    st.session_state.latest_volume = volume
    
    diff = pred_price - close_price
    signal = 'WAIT'
    
    if diff > st.session_state.threshold:
        signal = 'BUY'
        if len(st.session_state.plot_real_prices) > 5:
            sma5 = sum(list(st.session_state.plot_real_prices)[-5:]) / 5
            if close_price < sma5:
                signal = 'WAIT'
        if signal == 'BUY':
            execute_trade('LONG', close_price)
    elif diff < -st.session_state.threshold:
        signal = 'SELL'
        if len(st.session_state.plot_real_prices) > 5:
            sma5 = sum(list(st.session_state.plot_real_prices)[-5:]) / 5
            if close_price > sma5:
                signal = 'WAIT'
        if signal == 'SELL':
            execute_trade('SHORT', close_price)
    
    st.session_state.plot_real_prices.append(close_price)
    st.session_state.plot_pred_prices.append(pred_price)
    st.session_state.plot_signals.append(signal)
    candle_time = datetime.datetime.fromtimestamp(kline[0]/1000).strftime('%H:%M')
    st.session_state.plot_times.append(candle_time)
    
    if signal != 'WAIT':
        st.session_state.signals.append({
            'time': candle_time,
            'type': signal,
            'price': close_price
        })
    
    st.session_state.status = f"🟢 Обработана свеча в {candle_time}"

def execute_trade(action, price):
    if st.session_state.position == action:
        return
    fee_rate = 0.001
    
    if st.session_state.position != 'NONE':
        if st.session_state.position == 'LONG':
            profit_pct = (price - st.session_state.entry_price) / st.session_state.entry_price
        elif st.session_state.position == 'SHORT':
            profit_pct = (st.session_state.entry_price - price) / st.session_state.entry_price
        else:
            profit_pct = 0
        
        pnl = (st.session_state.trade_amount * profit_pct) - (st.session_state.trade_amount * fee_rate)
        st.session_state.balance += pnl
        
        if pnl > 0:
            st.session_state.winning_trades += 1
        else:
            st.session_state.losing_trades += 1
    
    st.session_state.position = action
    st.session_state.entry_price = price
    st.session_state.balance -= (st.session_state.trade_amount * fee_rate)
    st.session_state.trades_count += 1

def update_pnl():
    if st.session_state.position != 'NONE':
        try:
            url = "https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT"
            headers = {'User-Agent': 'Mozilla/5.0'}
            response = requests.get(url, timeout=5, headers=headers)
            if response.status_code == 200:
                current_price = float(response.json()['price'])
                if st.session_state.position == 'LONG':
                    st.session_state.unrealized_pnl = (current_price - st.session_state.entry_price) / st.session_state.entry_price * st.session_state.trade_amount
                elif st.session_state.position == 'SHORT':
                    st.session_state.unrealized_pnl = (st.session_state.entry_price - current_price) / st.session_state.entry_price * st.session_state.trade_amount
                
                if st.session_state.unrealized_pnl <= -st.session_state.stop_loss_usd:
                    st.session_state.balance += st.session_state.unrealized_pnl - (st.session_state.trade_amount * 0.001)
                    st.session_state.losing_trades += 1
                    st.session_state.position = 'NONE'
                    st.session_state.entry_price = 0.0
                    st.session_state.unrealized_pnl = 0.0
                    st.session_state.status = f"🚨 STOP-LOSS @ ${current_price}"
        except:
            pass

init_session_state()
poll_new_data()
update_pnl()

st.sidebar.title("⚙️ Управление")
st.session_state.learning_rate = st.sidebar.number_input("LR", 0.0001, 0.1, st.session_state.learning_rate, 0.0001, "%.4f")
st.session_state.threshold = st.sidebar.number_input("Порог (USD)", 1.0, 1000.0, st.session_state.threshold, 5.0)
st.session_state.stop_loss_usd = st.sidebar.number_input("Stop-Loss", 1.0, 500.0, st.session_state.stop_loss_usd, 1.0)
st.session_state.trade_amount = st.sidebar.number_input("Сделка (USD)", 10.0, 100000.0, st.session_state.trade_amount, 100.0)
st.session_state.max_points_on_plot = st.sidebar.slider("Свечей на графике", 10, 500, st.session_state.max_points_on_plot, 10)

st.sidebar.divider()
uptime = int(time.time() - st.session_state.start_time)
st.sidebar.metric("⏱ Аптайм", str(datetime.timedelta(seconds=uptime)))

st.title("🤖 LTC Pro Trading Bot")
st.markdown(f"**{TIME_WINDOW_MINUTES}м** | Нейроны: {NEURONS} | Фичи: price + volume")

c1, c2, c3, c4 = st.columns(4)
total_equity = st.session_state.balance + st.session_state.unrealized_pnl
c1.metric("Баланс", f"${st.session_state.balance:.2f}")
c2.metric("PnL", f"${st.session_state.unrealized_pnl:.2f}")
c3.metric("Капитал", f"${total_equity:.2f}", f"{total_equity - 1000.0:+.2f}")
pos_color = "green" if st.session_state.position == 'LONG' else "red" if st.session_state.position == 'SHORT' else "gray"
c4.markdown(f"**Позиция:** <span style='color:{pos_color}'>{st.session_state.position}</span> @ ${st.session_state.entry_price:.0f}", True)

s1, s2, s3, s4 = st.columns(4)
s1.metric("Сделок", st.session_state.trades_count)
s2.metric("Win", st.session_state.winning_trades)
s3.metric("Loss", st.session_state.losing_trades)
win_rate = (st.session_state.winning_trades / st.session_state.trades_count * 100) if st.session_state.trades_count > 0 else 0
s4.metric("Win Rate", f"{win_rate:.1f}%")

st.divider()
m1, m2, m3, m4 = st.columns(4)
m1.metric("Цена BTC", f"${st.session_state.latest_price:.2f}")
m2.metric("Объём", f"{st.session_state.latest_volume:.4f}")
m3.metric("Прогноз", f"${st.session_state.latest_pred:.2f}", f"{st.session_state.latest_pred - st.session_state.latest_price:+.2f}")
m4.metric("Loss", f"{st.session_state.loss:.6f}")

st.caption(f"**Статус:** {st.session_state.status}")
with st.expander("🔧 Debug"):
    st.text(f"API: {st.session_state.get('api_status', 'N/A')}")
    st.text(f"Error: {st.session_state.get('api_error', 'N/A')}")
    st.text(f"Hist: {len(st.session_state.price_history)} candles")

points = min(len(st.session_state.plot_real_prices), st.session_state.max_points_on_plot)
if points > 0:
    df = pd.DataFrame({
        'Время': list(st.session_state.plot_times)[-points:],
        'Цена': list(st.session_state.plot_real_prices)[-points:],
        'Прогноз': list(st.session_state.plot_pred_prices)[-points:]
    }).set_index('Время')
    
    st.subheader("График")
    st.line_chart(df, color=["#2962FF", "#FF0000"])
    
    st.subheader("Сигналы")
    for sig in reversed(st.session_state.signals[-10:]):
        emoji = "🟢" if sig['type'] == 'BUY' else "🔴"
        st.write(f"{emoji} **{sig['type']}** {sig['time']} @ ${sig['price']:.2f}")
else:
    st.info("Загрузка данных...")

time.sleep(POLL_INTERVAL)
st.rerun()
