import time
import requests
import threading

def block_cpu():
    print("Blocking CPU...")
    start = time.time()
    count = 0
    while time.time() - start < 10:
        count += 1
    print("Unblocking CPU")

t = threading.Thread(target=block_cpu)
t.start()

time.sleep(1)
print("Sending request...")
try:
    res = requests.get("http://localhost:8000/health", timeout=2)
    print("Response:", res.status_code)
except Exception as e:
    print("Error:", e)
