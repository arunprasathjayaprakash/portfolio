import google.oauth2.service_account as sc
import google.auth
from google.auth.transport.requests import Request
import requests
import json

def call_cloudrun(url,default="Hello"):

    #Run Your GCP setup for authentication and add Cloud Run permissions as needed
    credentials, _ = google.auth.default()
    auth_request = Request()
    credentials.refresh(auth_request)
    id_token = credentials.id_token

    payload = json.dumps({
        "msg": "data"
    })
    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {id_token}'}

    response = requests.request("POST", url, headers=headers, data=payload)

    return response.json()

if __name__ == "__main__":
    url = "https://cloud-trigger-32726136683.us-central1.run.app"
    call_cloudrun(url)