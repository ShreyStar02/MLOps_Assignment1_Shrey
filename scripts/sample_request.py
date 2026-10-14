"""Send sample requests to the running API (default http://localhost:8000)."""
import json
import random
import sys
import urllib.request

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 1
SAMPLE = {"age": 63, "sex": 1, "cp": 1, "trestbps": 145, "chol": 233, "fbs": 1,
          "restecg": 2, "thalach": 150, "exang": 0, "oldpeak": 2.3, "slope": 3,
          "ca": 0, "thal": 6}

for _ in range(N):
    payload = {**SAMPLE, "age": random.randint(30, 77), "chol": random.randint(150, 400),
               "thalach": random.randint(90, 195), "oldpeak": round(random.random() * 4, 1)}
    req = urllib.request.Request(f"{BASE}/predict", json.dumps(payload).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as r:  # noqa: S310
        print(r.read().decode())
