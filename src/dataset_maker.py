#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import base64
import io
import json
import logging
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import random
from typing import List, Dict
from random import randint

from PIL import Image

from src.kandinsky_api import retrieve_keys, KandinskyAPI
from src.common import DATA_PATH

logging.basicConfig(format='%(asctime)s [$(name)s] %(levelname)s %(message)s', level=logging.INFO)
logger = logging.getLogger('dataset_maker')


class DatasetMaker:
    def __init__(self,
                 output_dir: Path = DATA_PATH / 'retrieved',
                 visualize: bool = False, short_prompt=False):
        self.stats = defaultdict(dict)
        self.ids = defaultdict(list)
        self.logger = logging.getLogger('DatasetMaker')
        self.idx_generated = 0
        self.session_ts = datetime.now()
        self.session_name = self.session_ts.strftime("%m-%d-%Y/%H:%M:%S")
        self.output_dir = output_dir / self.session_name
        self.output_dir.mkdir(exist_ok=True, parents=True)
        self.visualize = visualize
        self.short_prompt = short_prompt

    def make_image(self,
                   image_class: str = 'balloon',
                   width: int = 320, height: int = 320,
                   attempts: int = 200, delay: int = 10):
        ts_start = datetime.now()
        api_key, secret_key = retrieve_keys()
        k_api = KandinskyAPI(url='https://api-key.fusionbrain.ai/',
                             api_key=api_key,
                             secret_key=secret_key)
        # image_class = 'goldfish'
        prompt = f'photo of {image_class}'
        if not self.short_prompt:
            prompt += f' ' \
                 f'in nature or ' \
                 f'in city or ' \
                 f'in interior or ' \
                 f'in the zoo or ' \
                 f'in the sky or ' \
                 f'in the wild or ' \
                 f'on the table'
        self.logger.info(f'sending request for "{image_class}"')
        request_id = k_api.generate(prompt=prompt,
                                    pipeline=k_api.get_pipeline(),
                                    images=1,
                                    width=width,
                                    height=height)
        self.ids[request_id].append(image_class)

        if 'requests' in self.stats[image_class]:
            self.stats[image_class]['requests'].append(request_id)
        else:
            self.stats[image_class]['requests'] = [request_id]

        self.logger.info(f'request for "{image_class}" sent successfully".\n'
                         f'Waiting for result making {attempts} attempts, with {delay} seconds delay.\n')
        result = k_api.check_generation(request_id=request_id,
                                        attempts=attempts,
                                        delay=delay)
        # todo: check status
        self.logger.info(f'result is ready. status: "{result['status']}"."')
        images = result['result']['files']
        image = Image.open(io.BytesIO(base64.b64decode(images[0])))
        retrieved_dir_path = self.output_dir
        if self.visualize:
            image.show()
        image.save(retrieved_dir_path / f'{image_class}_{self.idx_generated}.png')
        self.idx_generated += 1
        ts_end = datetime.now()
        self.logger.info(f'elapsed time: {ts_end - ts_start}')
        return image

    @staticmethod
    def propose_height_width(lo: int = 320, hi: int = 1024):
        ph = randint(lo, hi)
        pw = randint(int(ph * 0.8), int(ph * 1.2))
        pw = max(320, pw)
        return ph, pw

    def collect_images(self,
                       image_classes: Dict[str, List[str]],
                       limit=1000,
                       resolution=None):
        ts_start = datetime.now()
        logger.info('starting images collection')
        image_classes_keys = list(image_classes.keys())
        random.shuffle(image_classes_keys)
        for k in image_classes_keys:
            c = image_classes[k]
            if resolution is None:
                height, width = self.propose_height_width()
            else:
                height, width = resolution
            self.make_image(c[0], width=width, height=height)
            if self.idx_generated >= limit:
                break
        ts_end = datetime.now()
        self.logger.info(f'elapsed time (total): {ts_end - ts_start}')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # todo: add arguments
    parser.add_argument('--class-list-name',
                        # default='imagenet_1000',
                        default='goldfish_1',
                        help='class list name (without extension) or '
                             'classname and number of images e.g. goldfish_12')
    parser.add_argument('--resolution',
                        default='random',
                        help='use random or x-separated e.g. 320x320')
    args = parser.parse_args()
    class_list_name = args.class_list_name
    resolution = args.resolution
    width, height = map(int, resolution.split('x'))
    ds_maker = DatasetMaker(short_prompt=False)
    # ds_maker.make_image('goldfish')
    if class_list_name not in ['imagenet_1000']:
        class_name, n = class_list_name.split('_')
        n = int(n)
        image_classes = {str(i): [class_name] for i in range(n)}
        ds_maker.collect_images(image_classes, limit=n, resolution=(height, width))
    else:
        with open(DATA_PATH / f'{class_list_name}.json') as f:
            image_classes = json.load(f)
            ds_maker.collect_images(image_classes, limit=1000)
