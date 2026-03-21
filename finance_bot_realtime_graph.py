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
warnings.filterwarnings('ignore')

print("=== Жидкая нейросеть (LTC): ТРУ-РЕАЛТАЙМ + ОБУЧЕНИЕ НА ЛЕТУ ===")

# --- НАСТРОЙКИ ---
NEURONS = 64 # Увеличили в 4 раза! (было 16)
SEQ_LENGTH = 50 
MAX_POINTS_ON_PLOT = 100 
LEARNING_RATE = 0.002 # Снизили в 25 раз, чтобы прогноз был плавным и не прыгал (убрали колебания)
MODEL_PATH = "ltc_realtime_model.pth" # Файл для сохранения "мозгов"

class RealtimeLTC(nn.Module):
    def __init__(self):
        super().__init__()
        wiring = AutoNCP(units=NEURONS, output_size=1)
        self.ltc = LTC(input_size=1, units=wiring, batch_first=True)

    def forward(self, x):
        out, _ = self.ltc(x)
        return out[:, -1, :]

import os
model = RealtimeLTC()

# Загружаем сохраненные "мозги", если они есть (сохранение после рестарта)
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    print(f"🧠 Успешно загружены старые 'мозги' из файла {MODEL_PATH}!")
else:
    print("🧠 Сохраненных мозгов нет, начинаем обучение с чистого листа.")

model.train() # Включаем режим тренировки!

# Добавляем оптимизатор и функцию ошибки для обучения на лету
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
criterion = nn.MSELoss()

scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(np.array([[50000], [100000]])) # Чуть сузили диапазон для точности

price_history = [] 
plot_real_prices = deque(maxlen=MAX_POINTS_ON_PLOT)
plot_pred_prices = deque(maxlen=MAX_POINTS_ON_PLOT)
ticks_counter = 0 # Счетчик для сохранения модели

def on_message(ws, message):
    global price_history, ticks_counter
    try:
        data = json.loads(message)
        price = float(data['p'])
        
        scaled_price = scaler.transform([[price]])[0][0]
        price_history.append([scaled_price])
        
        # Нам нужен буфер на 1 больше, чтобы было с чем сравнивать (прошлое vs текущее)
        if len(price_history) > SEQ_LENGTH + 1:
            price_history.pop(0)
            
        if len(price_history) == SEQ_LENGTH + 1:
            # 1. ОБУЧЕНИЕ НА ЛЕТУ (Online Learning)
            X_train = torch.tensor([price_history[:-1]], dtype=torch.float32)
            y_true = torch.tensor([[price_history[-1]]], dtype=torch.float32)
            
            optimizer.zero_grad()
            y_pred = model(X_train)
            loss = criterion(y_pred, y_true)
            loss.backward()
            optimizer.step() # Обновляем "мозг" нейросети!
            
            # Сохраняем "мозги" каждые 100 тиков
            ticks_counter += 1
            if ticks_counter % 100 == 0:
                torch.save(model.state_dict(), MODEL_PATH)
            
            # 2. ПРЕДСКАЗАНИЕ БУДУЩЕГО (на шаг вперед)
            X_test = torch.tensor([price_history[1:]], dtype=torch.float32)
            with torch.no_grad():
                future_pred_scaled = model(X_test).numpy()
            
            pred_price = scaler.inverse_transform(future_pred_scaled)[0][0]
            
            plot_real_prices.append(price)
            plot_pred_prices.append(pred_price)
            
            print(f"[Учится] Цена: {price:.2f}$ | Прогноз: {pred_price:.2f}$ | Ошибка: {loss.item():.7f}", end='\r', flush=True)
        else:
            print(f"Сбор данных для памяти... ({len(price_history)}/{SEQ_LENGTH+1})", end='\r', flush=True)
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
        print("\n✅ Подключено! Начинаем сбор данных (50 тиков) для старта...", flush=True)
        is_first_connect = False
    else:
        print("\n✅ Успешно переподключено! Память и знания нейросети сохранены. Продолжаем...", flush=True)

def start_websocket():
    socket_url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    while True:
        ws = websocket.WebSocketApp(socket_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
        # Отправляем пинг каждую минуту, чтобы Binance не закрывал соединение
        ws.run_forever(ping_interval=60, ping_timeout=10)
        print("\nПереподключение через 2 секунды...", flush=True)
        time.sleep(2)

ws_thread = threading.Thread(target=start_websocket, daemon=True)
ws_thread.start()

fig, ax = plt.subplots(figsize=(10, 6))
fig.canvas.manager.set_window_title('LTC Online Learning')
ax.set_title("Жидкая Нейросеть (LTC): Обучение на лету (Online Learning)")
ax.set_xlabel("Последние тики (сделки)")
ax.set_ylabel("Цена (USD)")

line_real, = ax.plot([], [], label='Реальная цена', color='blue', linewidth=2)
line_pred, = ax.plot([], [], label='Прогноз LTC', color='red', linestyle='--', linewidth=2)
ax.legend()
ax.grid(True)

def update_plot(frame):
    if len(plot_real_prices) > 0:
        x_data = list(range(len(plot_real_prices)))
        line_real.set_data(x_data, list(plot_real_prices))
        line_pred.set_data(x_data, list(plot_pred_prices))
        
        ax.set_xlim(0, max(MAX_POINTS_ON_PLOT, len(plot_real_prices)))
        
        # Сильное приближение графика (вплоть до десятков центов)
        min_y = min(min(plot_real_prices), min(plot_pred_prices)) - 0.2
        max_y = max(max(plot_real_prices), max(plot_pred_prices)) + 0.2
        ax.set_ylim(min_y, max_y)
        
        # Запрещаем научный формат (типа 7.06e4) на оси Y
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')
    return line_real, line_pred

ani = FuncAnimation(fig, update_plot, interval=100, cache_frame_data=False)
plt.show()
