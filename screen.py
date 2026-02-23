import os
import time
import datetime
from pathlib import Path

import requests
import mss

# ================== CONFIGURACIÓN ==================

BOT_TOKEN = "6846433896:AAGvDCxu-fsbYSHSa1KrMCm0HDV6uIA2fLw"      # Ejemplo: "123456789:AA...."
CHAT_ID = "6157582845"         # Ejemplo: "123456789" o "-1001234567890" para grupos

# Intervalo entre rondas de screenshots (en segundos)
SCREENSHOT_INTERVAL_SECONDS = 60 * 2

# ============================================


def take_screenshots_all_monitors() -> list[bytes]:
    """
    Toma un screenshot separado de cada monitor físico.
    Devuelve una lista de datos de imagen en bytes.
    """
    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    screenshot_data: list[bytes] = []
    print(f"[INFO] Tomando capturas de pantalla a las {now_str}...")
    with mss.mss() as sct:
        # sct.monitors[0] suele ser el escritorio virtual completo.
        # Los monitores físicos empiezan en 1: [1], [2], ...
        num_monitors = len(sct.monitors) - 1
        if num_monitors <= 0:
            # Por seguridad, si no detecta monitores físicos, al menos uno global
            img = sct.grab(sct.monitors[0])
            screenshot_data.append(mss.tools.to_png(img.rgb, img.size))
            return screenshot_data

        for i in range(1, len(sct.monitors)):
            img = sct.grab(sct.monitors[i])
            screenshot_data.append(mss.tools.to_png(img.rgb, img.size))

    return screenshot_data


def send_to_telegram(image_data: bytes, monitor_index: int) -> None:
    """Envía una imagen a tu chat de Telegram."""
    url = f"https://api.telegram.org/bot{BOT_TOKEN}/sendPhoto"

    now_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"screenshot_monitor{monitor_index}_{now_str}.png"
    
    files = {"photo": (filename, image_data, "image/png")}
    data = {"chat_id": CHAT_ID}
    resp = requests.post(url, data=data, files=files, timeout=30)

    if not resp.ok:
        print(f"[ERROR] Falló el envío a Telegram: {resp.status_code} - {resp.text}")


def main():
    if not BOT_TOKEN or BOT_TOKEN == "TU_BOT_TOKEN_AQUI":
        print("Configurá BOT_TOKEN y CHAT_ID antes de ejecutar el script.")
        return

    print("Iniciando captura automática de pantalla (todos los monitores).")
    print(f"Intervalo: {SCREENSHOT_INTERVAL_SECONDS} segundos.")
    print("Presioná Ctrl+C para detener.\n")

    try:
        while True:
            try:
                screenshots = take_screenshots_all_monitors()
                timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                print(f"[{timestamp}] Capturas tomadas:")

                for i, image_data in enumerate(screenshots, 1):
                    print(f"  - Monitor {i}")
                    send_to_telegram(image_data, i)
                    print("    Enviada a Telegram.")

                print()

            except Exception as e:
                print(f"[ERROR] Ocurrió un problema: {e}")
                print("Reintentando en 10 segundos...\n")
                time.sleep(10)
                continue

            time.sleep(SCREENSHOT_INTERVAL_SECONDS)

    except KeyboardInterrupt:
        print("\nDetenido por el usuario. Fin.")


if __name__ == "__main__":
    main()
