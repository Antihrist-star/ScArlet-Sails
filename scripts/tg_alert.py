
import requests
import json
import sys
import os
from datetime import datetime

def send_telegram_message(bot_token, chat_id, message):
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": message,
        "parse_mode": "Markdown"
    }
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()  # Raise an exception for HTTP errors
        print(f"Telegram message sent successfully: {message}")
        return True
    except requests.exceptions.RequestException as e:
        log_error(f"Failed to send Telegram message: {e}")
        return False

def log_error(error_message):
    log_dir = "checks"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "tg_err.log")
    with open(log_file, "a") as f:
        f.write(f"[{datetime.now()}] {error_message}\n")
    print(f"Error logged: {error_message}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python tg_alert.py \"Your message here\"")
        sys.exit(1)

    message_to_send = sys.argv[1]

    config_path = "configs/telegram.json"
    if not os.path.exists(config_path):
        log_error(f"Telegram config file not found at {config_path}")
        sys.exit(1)

    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        bot_token = config["bot_token"]
        chat_id = config["chat_id"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError) as e:
        log_error(f"Error loading Telegram configuration: {e}")
        sys.exit(1)

    send_telegram_message(bot_token, chat_id, message_to_send)

