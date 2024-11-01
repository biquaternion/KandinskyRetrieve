#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from typing import List

from src.project_const import DATA_PATH


class ImagenetMapping:
    def __init__(self):
        self.classes_path = DATA_PATH / 'imagenet/classes.txt'
        self.classes = self.get_classes()

    def get_classes(self) -> List[List[str]]:
        with open(self.classes_path) as f:
            for line in f:
                self.classes.append(line.strip().split(', '))
        return self.classes

    def


if __name__ == '__main__':
    with open(DATA_PATH / 'imagenet_1000_classes.txt', 'r') as f:
        for line in f.readlines():
            print(line.strip().split(', '))
