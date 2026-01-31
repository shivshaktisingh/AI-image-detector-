import requests

API_KEY = "LupJX6wuZ1K9kNoTxZCvuJMvjEFATCLN0tb6vPTIUoc"  # replace with value in your .env
URL = "http://127.0.0.1:5000/api/detect"
IMAGE_PATH = "2.jpg"  # put your test image in project root or change path

headers = {"x-api-key": API_KEY}

with open(IMAGE_PATH, "rb") as f:
    files = {"image": f}
    r = requests.post(URL, headers=headers, files=files)

print("Status:", r.status_code)
try:
    print("Response JSON:", r.json())
except Exception:
    print("Response text:", r.text)
