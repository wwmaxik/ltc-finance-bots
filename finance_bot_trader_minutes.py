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

# --- НАСТРОЙКИ ---
TIME_WINDOW_MINUTES = 1.0  # 1.0 = 1 минута, 5.0 = 5 минут (можно менять)
TIME_WINDOW_SEC = TIME_WINDOW_MINUTES * 60.0

print(f"=== Жидкая нейросеть (LTC): ТОРГОВЫЙ БОТ (Таймфрейм: {TIME_WINDOW_MINUTES} мин) ===")

NEURONS = 64
SEQ_LENGTH = 50 
MAX_POINTS_ON_PLOT = 50 # Покажем 50 свечей на графике
LEARNING_RATE = 0.002 
MODEL_PATH = f"ltc_trader_model_{TIME_WINDOW_MINUTES}m.pth" # Отдельный файл для каждого таймфрейма

FEE_USD = 2.0          
MIN_PROFIT = 5.0       # Для минутного графика ожидаем прибыль побольше
THRESHOLD = FEE_USD + MIN_PROFIT 

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
plot_signals = deque(maxlen=MAX_POINTS_ON_PLOT) 

ticks_counter = 0 
last_process_time = 0
current_candle_prices = [] # Собираем все цены внутри минуты для вычисления средней/закрытия

def on_message(ws, message):
    global price_history, ticks_counter, last_process_time, current_candle_prices
    try:
        data = json.loads(message)
        current_price = float(data['p'])
        current_time = time.time()
        
        # Собираем цены для текущей свечи
        current_candle_prices.append(current_price)
        
        # Если скрипт только запустился, инициализируем таймер
        if last_process_time == 0:
            last_process_time = current_time
            print(f"⏳ Сбор первой свечи ({int(TIME_WINDOW_MINUTES*60)} сек)...", flush=True)
            
        # Проверяем, прошло ли нужное время (например, 1 минута = 60 секунд)
        elapsed_time = current_time - last_process_time
        
        if elapsed_time >= TIME_WINDOW_SEC:
            last_process_time = current_time
            
            # Берем цену закрытия (последнюю цену в этой минуте)
            candle_close_price = current_candle_prices[-1]
            current_candle_prices.clear() # Очищаем буфер для следующей минуты
            
            scaled_price = scaler.transform([[candle_close_price]])[0][0]
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
                if ticks_counter % 10 == 0: # Сохраняем чаще (каждые 10 минут)
                    torch.save(model.state_dict(), MODEL_PATH)
                
                # 2. ПРЕДСКАЗАНИЕ
                X_test = torch.tensor([price_history[1:]], dtype=torch.float32)
                with torch.no_grad():
                    future_pred_scaled = model(X_test).numpy()
                
                pred_price = scaler.inverse_transform(future_pred_scaled)[0][0]
                
                diff = pred_price - candle_close_price
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
                
                plot_real_prices.append(candle_close_price)
                plot_pred_prices.append(pred_price)
                plot_signals.append(signal)
                
                print(f"[{time.strftime('%H:%M:%S')}] Цена закрытия свечи: {candle_close_price:.2f}$ | Прогноз на след. свечу: {pred_price:.2f}$ | Разница: {diff:+.2f}$ | Сигнал: {signal_text}", flush=True)
            else:
                print(f"⏳ Сбор свечей... ({len(price_history)}/{SEQ_LENGTH+1})", flush=True)
        else:
            # Показываем прогресс-бар до следующей свечи в консоли (перезаписывая строку)
            progress = int((elapsed_time / TIME_WINDOW_SEC) * 100)
            print(f"Формирование текущей {TIME_WINDOW_MINUTES}м свечи: [{progress}%]", end='\r', flush=True)
            
    except Exception as e:
        print(f"\nОшибка в on_message: {e}", flush=True)

def on_error(ws, error):
    pass
def on_close(ws, close_status_code, close_msg):
    print("\nСоединение с биржей закрыто.")

import requests

def fetch_historical_candles():
    print(f"\n⚡ Загрузка истории (последние {SEQ_LENGTH} свечей по {TIME_WINDOW_MINUTES}м) через Binance REST API...")
    # Преобразуем минуты в формат Binance (1m, 3m, 5m, 15m)
    interval = f"{int(TIME_WINDOW_MINUTES)}m"
    url = f"https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval={interval}&limit={SEQ_LENGTH+1}"
    
    response = requests.get(url)
    if response.status_code == 200:
        data = response.json()
        for kline in data:
            # kline[4] это цена закрытия (Close price)
            close_price = float(kline[4])
            scaled_price = scaler.transform([[close_price]])[0][0]
            price_history.append([scaled_price])
        print(f"✅ История загружена! Память заполнена ({len(price_history)}/{SEQ_LENGTH+1}).")
    else:
        print("❌ Ошибка при загрузке истории. Придется собирать данные вручную.")

is_first_connect = True
def on_open(ws):
    global is_first_connect
    if is_first_connect:
        print(f"\n✅ Подключено к WebSocket! Запущен режим {TIME_WINDOW_MINUTES}м.", flush=True)
        fetch_historical_candles() # Загружаем историю моментально!
        is_first_connect = False
    else:
        print("\n✅ Переподключено. Продолжаем...", flush=True)

def start_websocket():
    socket_url = "wss://stream.binance.com:9443/ws/btcusdt@trade"
    while True:
        ws = websocket.WebSocketApp(socket_url, on_open=on_open, on_message=on_message, on_error=on_error, on_close=on_close)
        ws.run_forever(ping_interval=60, ping_timeout=10)
        time.sleep(2)

ws_thread = threading.Thread(target=start_websocket, daemon=True)
ws_thread.start()

# --- ОТРИСОВКА ГРАФИКА С СИГНАЛАМИ ---
fig, ax = plt.subplots(figsize=(10, 6))
fig.canvas.manager.set_window_title(f'LTC {TIME_WINDOW_MINUTES}m Trader')
ax.set_title(f"LTC Торговый Бот (Свечи по {TIME_WINDOW_MINUTES} мин)")
ax.set_xlabel("Последние свечи")
ax.set_ylabel("Цена (USD)")

line_real, = ax.plot([], [], label='Реальная цена (Close)', color='blue', linewidth=2)
line_pred, = ax.plot([], [], label='Прогноз (на 1 свечу вперед)', color='red', linestyle='--', linewidth=2)

line_buy, = ax.plot([], [], marker='^', color='green', markersize=10, linestyle='None', label='Сигнал BUY')
line_sell, = ax.plot([], [], marker='v', color='red', markersize=10, linestyle='None', label='Сигнал SELL')

ax.legend()
ax.grid(True)

def update_plot(frame):
    if len(plot_real_prices) > 0:
        x_data = list(range(len(plot_real_prices)))
        line_real.set_data(x_data, list(plot_real_prices))
        line_pred.set_data(x_data, list(plot_pred_prices))
        
        x_buy = [i for i, s in enumerate(plot_signals) if s == 'BUY']
        y_buy = [plot_real_prices[i] for i in x_buy]
        line_buy.set_data(x_buy, y_buy)
        
        x_sell = [i for i, s in enumerate(plot_signals) if s == 'SELL']
        y_sell = [plot_real_prices[i] for i in x_sell]
        line_sell.set_data(x_sell, y_sell)
        
        ax.set_xlim(0, max(MAX_POINTS_ON_PLOT, len(plot_real_prices)))
        min_y = min(min(plot_real_prices), min(plot_pred_prices)) - 10.0
        max_y = max(max(plot_real_prices), max(plot_pred_prices)) + 10.0
        ax.set_ylim(min_y, max_y)
        ax.ticklabel_format(useOffset=False, style='plain', axis='y')
        
    return line_real, line_pred, line_buy, line_sell

ani = FuncAnimation(fig, update_plot, interval=1000, cache_frame_data=False) # График обновляем реже (раз в секунду)
plt.show()
