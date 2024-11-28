import os
import requests
import base64
import json
# Configuration
API_KEY = "430a2b9d48af48d2a28c5b5f56ccd653"
headers = {
    "Content-Type": "application/json",
    "api-key": API_KEY,
}

# Payload for the request
payload = {
  "messages": [
    {
      "role": "system",
      "content": [
        {
          "type": "text",
          "text": "You are a helpful AI."
        }
      ]
    },
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": "Translate English to Hausa.\n#English#\nI'd like to go to Shanghai.\n#Hausa#\n"
        }
      ]
    }
  ],
  "temperature": 0.1,
  "top_p": 0.1,
  "max_tokens": 800
}

ENDPOINT = "https://gpt4frankfurt.openai.azure.com/openai/deployments/gpt-4o/chat/completions?api-version=2024-02-15-preview"

#基础访问gpt的代码
def getgpt(language = 'en', title = '', tags = [], times = 10):
    # Send request
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload)
        response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
    except requests.RequestException as e:
        raise e

    # Handle the response as needed (e.g., print or process)
    print(response.json())

if __name__=='__main__':
    getgpt()