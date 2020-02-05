import requests

response = requests.get('http://192.168.0.106:5000')
print(response.status_code)
