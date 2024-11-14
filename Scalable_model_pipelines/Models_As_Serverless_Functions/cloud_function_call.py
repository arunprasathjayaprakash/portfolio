import requests

def call_cloudrun(url,default='Hello'):

    #POST the url for a resonse
    results = requests.post(url,
                             json={'msg':default})

    return results.json()

if __name__ == "__main__":
    url = 'https://us-central1-manifest-glyph-441000-g2.cloudfunctions.net/Cloud_trigger'
    call_cloudrun(url)