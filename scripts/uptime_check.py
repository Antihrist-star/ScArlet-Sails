import requests
import time
from datetime import datetime
import os

repo_url = "https://github.com/Antihrist-star/scarlet-sails"

def check_uptime():
    try:
        r = requests.get(repo_url)
        status_code = r.status_code
        message = f"{datetime.now()}: {repo_url} status: {status_code}"
    except requests.exceptions.RequestException as e:
        status_code = "Error"
        message = f"{datetime.now()}: {repo_url} error: {e}"

    log_file = "checks/uptime.log"
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    with open(log_file, "a") as f:
        f.write(message + "\n")
    print(message)

if __name__ == "__main__":
    check_uptime()

