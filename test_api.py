import requests

url = "http://127.0.0.1:8000/explain"
data = {"code": "def factorial(n): return 1 if n==0 else n*factorial(n-1)"}
r = requests.post(url, json=data)
print(r.json())
