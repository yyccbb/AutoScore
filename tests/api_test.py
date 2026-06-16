import os
import sys
import requests
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils.env import load_env

load_env()

api_key = os.environ['OPENROUTER_API_KEY']
base_url = "https://openrouter.ai/api/v1/chat/completions"

def test_grader_model(model_name):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        {
            "role": "user",
            "content": "Introduce yourself"
        }
    ]

    response = requests.post(
        url=base_url,
        headers=headers,
        json={
            "model": model_name,
            "messages": messages
        }
    )

    data = response.json()
    return data["choices"][0]["message"]["content"]

def test_reflector_model(model_name):
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    messages = [
        {
            "role": "user",
            "content": "Introduce yourself"
        }
    ]

    response = requests.post(
        url=base_url,
        headers=headers,
        json={
            "model": model_name,
            "messages": messages,
            "reasoning": {
                "enabled": True
            }
        }
    )

    data = response.json()
    return data["choices"][0]["message"]["content"]
    
def main():
    grader_model = "qwen/qwen-2.5-72b-instruct"
    reflector_model = "deepseek/deepseek-r1"
    # print(test_grader_model(grader_model))
    print(test_reflector_model(reflector_model))

if __name__ == "__main__":
    main()