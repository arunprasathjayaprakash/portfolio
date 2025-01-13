import google.oauth2.service_account as sc
import google.auth
from google.auth.transport.requests import Request
import requests
import json

def call_cloudrun(url,):

    #Run Your GCP setup for authentication and add Cloud Run permissions as needed
    credentials, _ = google.auth.default()
    auth_request = Request()
    credentials.refresh(auth_request)
    id_token = credentials.id_token

    payload = {"0": 0, "1": 0, "2": 0, "3": 0, "4": 0, "5": 0, "6": 0, "7": 0, "8": 0, "9": 0}

    headers = {
        'Content-Type': 'application/json',
        'Authorization': f'Bearer {id_token}'}

    response = requests.request("POST", url, headers=headers, data=json.dumps(payload))

    return json.loads(response.content)

if __name__ == "__main__":
    url = "CLOUD FUNCTION DEPLOYED URL"
    call_cloudrun(url)