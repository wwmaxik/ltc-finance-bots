import json
import websocket
import torch
import torch.nn as nn
import numpy as np
from ncps.torch import LTC
from ncps.wirings import AutoNCP
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
from collections import deque
import warnings
import time
import os
warnings.filterwarnings('ignore')

print("=== Жидкая нейросеть (LTC): ТОРГОВЫЙ БОТ (Сек. таймфрейм + Сигналы) ===")

# --- НАСТРОЙКИ Нейросети ---
NEURONS = 64
SEQ_LENGTH = 50 
MAX_POINTS_ON_PLOT = 100 
LEARNING_RATE = 0.002 
MODEL_PATH = "ltc_trader_model.pth"

# --- НАСТРОЙКИ Торговой стратегии (ПОИНТ 1 и 2) ---
TIME_WINDOW_SEC = 1.0  # (Поинт 1) Срезы цены ровно раз в 1 секунду (сглаживает шум от тиков)
FEE_USD = 2.0          # Условная комиссия биржи в долларах (занижено для демо)
MIN_PROFIT = 2.0       # Сколько минимум хотим заработать с одной сделки
THRESHOLD = FEE_USD + MIN_PROFIT # (Поинт 2) Порог реакции (разница должна быть > $4.0)

class RealtimeLTC(nn.Module):
    def __init__(self):
        super().__init__()
        wiring = AutoNCP(units=NEURONS, output_size=1)
        self.ltc = LTC(input_size=1, units=wiring, batch_first=True)

    def forward(self, x):
        out, _ = self.ltc(x)
        return out[:, -1, :]

model = RealtimeLTC()

if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    print(f"🧠 Успешно загружены 'мозги' из {MODEL_PATH}!")
else:
    print("🧠 Начинаем обучение торгового бота с чистого листа.")

model.train() 
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.array([[50000], [100000]])) 

price_history = [] 
plot_real_prices = deque(maxlen=MAX_POINTS_ON_PLOT)
plot_pred_prices = deque(maxlen=MAX_POINTS_ON_PLOT)
plot_signals = deque(maxlen=MAX_POINTS_ON_PLOT) # Для хранения сигналов BUY/SELL

ticks_counter = 0 
last_process_time = 0

def on_message(ws, message):
    global price_history, ticks_counter, last_process_time
    try:
        data = json.loads(message)
        current_price = float(data['p'])
        current_time = time.time()
        
        # ПОИНТ 1: Проверяем, прошла ли 1 секунда с прошлого среза
        if current_time - last_process_time >= TIME_WINDOW_SEC:
            last_process_time = current_time
            
            scaled_price = scaler.transform([[current_price]])[0][0]
            price_history.append([scaled_price])
            
            if len(price_history) > SEQ_LENGTH + 1:
                price_history.pop(0)
                
            if len(price_history) == SEQ_LENGTH + 1:
                # 1. ОБУЧЕНИЕ НА ЛЕТУ 
                X_train = torch.tensor([price_history[:-1]], dtype=torch.float32)
                y_true = torch.tensor([[price_history[-1]]], dtype=torch.float32)
                
                optimizer.zero_grad()
                y_pred = model(X_train)
                loss = criterion(y_pred, y_true)
                loss.backward()
                optimizer.step()
                
                ticks_counter += 1
                if ticks_counter % 100 == 0:
                    torch.save(model.state_dict(), MODEL_PATH)
                
                # 2. ПРЕДСКАЗАНИЕ
                X_test = torch.tensor([price_history[1:]], dtype=torch.float32)
                with torch.no_grad():
                    future_pred_scaled = model(X_test).numpy()
                
                pred_price = scaler.inverse_transform(future_pred_scaled)[0][0]
                
                # ПОИНТ 2: ЛОГИКА ТОРГОВЫХ СИГНАЛОВ
                diff = pred_price - current_price
                signal = None
                signal_text = ""
                
                if diff > THRESHOLD:
                    signal = 'BUY'
                    signal_text = "🟢 БЕРЕМ В LONG (BUY)"
                elif diff < -THRESHOLD:
                    signal = 'SELL'
                    signal_text = "🔴 ПРОДАЕМ (SELL)  "
                else:
                    signal = 'WAIT'
                    signal_text = "⚪ ЖДЕМ (WAIT)      "
                
                plot_real_prices.append(current_price)
                plot_pred_prices.append(pred_price)
                plot_signals.append(signal)
                
                print(f"[1 СЕК] Цена: {current_price:.2f}$ | Прогноз: {pred_price:.2f}$ | Разница: {diff:+.2f}$ | Сигнал: {signal_text}", end='\r', flush=True)
            else:
                print(f"Сбор секундных фреймов... ({len(price_history)}/{SEQ_LENGTH+1})", end='\r', flush=True)
    except Exception as e:
        print(f"\nОшибка в on_message: {e}", flush=True)

def on_error(ws, error):
    pass
def on_close(ws, close_status_code, close_msg):
    print("\nСоединение с биржей закрыто.")

is_first_connect = True
def on_open(ws):
    global is_first_connect
    if is_first_connect:
        print("\n✅ Подключено! Ждем накопления 50 секундных окон...", flush=True)
        is_first_connect = False
    else:
        print("\n✅ Переподключено. Продолжаем торговлю...", flush=True)

def start_websocket():
    socket_url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    while True:
        ws = websocket.WebSocketApp(socket_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
        ws.run_forever(ping_interval=60, ping_timeout=10)
        print("\nПереподключение через 2 секунды...", flush=True)
        time.sleep(2)

ws_thread = threading.Thread(target=start_websocket, daemon=True)
ws_thread.start()

# --- ОТРИСОВКА ГРАФИКА С СИГНАЛАМИ ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.canvas.manager.set_window_title('LTC HFT Trading Bot')
ax.set_title("LTC Торговый Бот (1-секундные свечи + Сигналы Входа)")
ax.set_xlabel("Последние секунды")
ax.set_ylabel("Цена (USD)")

line_real, = ax.plot([], [], label='Реальная цена', color='blue', linewidth=2)
line_pred, = ax.plot([], [], label='Прогноз LTC', color='red', linestyle='--', linewidth=2)

# Маркеры для сигналов покупки и продажи
line_buy, = ax.plot([], [], marker='^', color='green', markersize=10, linestyle='None', label='Сигнал BUY')
line_sell, = ax.plot([], [], marker='v', color='red', markersize=10, linestyle='None', label='Сигнал SELL')

ax.legend()
ax.grid(True)

def update_plot(frame):
    if len(plot_real_prices) > 0:
        x_data = list(range(len(plot_real_prices)))
        line_real.set_data(x_data, list(plot_real_prices))
        line_pred.set_data(x_data, list(plot_pred_prices))
        
        # Отрисовка маркеров покупки/продажи на графике
        x_buy = [i for i, s in enumerate(plot_signals) if s == 'BUY']
        y_buy = [plot_real_prices[i] for i in x_buy]
        line_buy.set_data(x_buy, y_buy)
        
        x_sell = [i for i, s in enumerate(plot_signals) if s == 'SELL']
        y_sell = [plot_real_prices[i] for i in x_sell]
        line_sell.set_data(x_sell, y_sell)
        
        ax.set_xlim(0, max(MAX_POINTS_ON_PLOT, len(plot_real_prices)))
        min_y = min(min(plot_real_prices), min(plot_pred_prices)) - 1.0
        max_y = max(max(plot_real_prices), max(plot_pred_prices)) + 1.0
        ax.set_ylim(min_y, max_y)
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')
        
    return line_real, line_pred, line_buy, line_sell

ani = FuncAnimation(fig, update_plot, interval=100, cache_frame_data=False)
plt.show()
