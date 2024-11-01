#!/usr/bin/env python3
# -*- coding: utf-8

import base64
import io
import json
import logging
import os
import time
from typing import Dict, Tuple

import requests
from PIL import Image

from src.project_const import DATA_PATH


def retrieve_keys() -> Tuple[str, str]:
    api_key = os.environ['KANDINSKY_API_KEY']
    secret_key = os.environ['KANDINSKY_SECRET_KEY']
    return api_key, secret_key


class KandinskyAPI:
    def __init__(self, url: str, api_key: str, secret_key: str):
        self.url = url
        self.api_key = api_key
        self.secret_key = secret_key
        self.auth_headers = {
            'X-Key': f'Key {self.api_key}',
            'X-Secret': f'Secret {self.secret_key}'
        }
        self.logger = logging.getLogger('KandinskyAPI')

    def get_model(self):
        model_path = 'key/api/v1/models'
        response = requests.get(url=self.url + model_path,
                                headers=self.auth_headers)
        data = response.json()
        return data[0]['id']

    def generate(self, prompt: str, model: Dict, images=1, width=1024, height=1024):
        params = {
            'type': 'GENERATE',
            'num_images': images,
            'width': width,
            'height': height,
            'generateParams': {
                'query': f'{prompt}'
            }
        }
        data = {
            'model_id': (None, model),
            'params': (None, json.dumps(params), 'application/json'),
        }
        response = requests.post(url=self.url + 'key/api/v1/text2image/run',
                                 headers=self.auth_headers, files=data)
        data = response.json()
        return data['uuid']

    def check_generation(self, request_id: str, attempts: int = 10, delay: int = 10):
        while attempts > 0:
            response = requests.get(url=self.url + 'key/api/v1/text2image/status/' + request_id,
                                    headers=self.auth_headers)
            data = response.json()
            if data['status'] == 'DONE':
                return data
            else:
                self.logger.info(f'request status: {data['status']}')

            attempts -= 1
            time.sleep(delay)
        return data


if __name__ == '__main__':
    api_key, secret_key = retrieve_keys()
    image_class = 'balloon'
    r = KandinskyAPI(url='https://api-key.fusionbrain.ai/',
                     api_key=api_key,
                     secret_key=secret_key)
    model = r.get_model()
    request_id = r.generate(prompt=f'photo of {image_class} '
                                   f'in nature or '
                                   f'in city or '
                                   f'in interior or '
                                   f'in aquarium or '
                                   f'in the zoo or '
                                   f'in the sky or '
                                   f'in the wild or '
                                   f'on the table',
                            model=model, images=1,
                            width=320, height=320)
    print(request_id)
    result = r.check_generation(request_id=request_id,
                                attempts=200,
                                delay=10)
    result = result['images']
    print(result)
    image = Image.open(io.BytesIO(base64.b64decode(result[0])))
    retrieved_dir_path = DATA_PATH / 'retrieved'
    image.show()
    image.save(retrieved_dir_path / f'{image_class}.png')
