import requests,time
import pandas as pd
from datetime import datetime

def date_to_timestamp(date):
    return int(datetime.strptime(date, "%Y-%m-%d").timestamp())

start = date_to_timestamp("2025-01-01")
end = date_to_timestamp("2026-01-01")
url = "https://priceapi.moneycontrol.com/techCharts/indianMarket/index/history?symbol=in%3BNSX&resolution=60&from="+str(start)+"&to="+str(end)+"&countback=1115&currencyCode=INR"

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
}

response = requests.get(url, headers=headers)
print(f"Status Code: {response.status_code}")
print(f"Content-Type: {response.headers.get('Content-Type')}")

try:
    data = response.json()
    print(data)
except requests.exceptions.JSONDecodeError:
    print("Failed to decode JSON. Response text:")
    print(response.text)
