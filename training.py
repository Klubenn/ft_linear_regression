#! /usr/bin/env python3

from json import dumps
from pathlib import Path
from typing import Tuple

import numpy as np

LEARNING_RATE = 1
ITERATIONS = 50


def estimated_price(t0, t1, km):
    return t1 * km + t0


def calculate_square_error(t0, t1, km, price) -> float:
    sum_ = sum([(estimated_price(t0, t1, x) - y) ** 2 for x, y in zip(km, price)])
    return (1 / (2 * len(km))) * sum_


def get_tmp_t0(t0, t1, km, price):
    size = len(km)
    sum_ = sum([estimated_price(t0, t1, km_i) - price_i for km_i, price_i in zip(km, price)])
    return (LEARNING_RATE / size) * sum_


def get_tmp_t1(t0, t1, km, price):
    size = len(km)
    sum_ = sum([(estimated_price(t0, t1, km_i) - price_i) * km_i for km_i, price_i in zip(km, price)])
    return (LEARNING_RATE / size) * sum_


def train_model(t0: float, t1: float, km: list, price: list) -> Tuple[float, float]:
    for _ in range(ITERATIONS):
        tmp_t0 = get_tmp_t0(t0, t1, km, price)
        tmp_t1 = get_tmp_t1(t0, t1, km, price)
        print(t0, t1)
        t0 = tmp_t0
        t1 = tmp_t1
    return t0, t1


def write_to_file(t0, t1) -> None:
    data = {'theta0': t0, 'theta1': t1}
    p = Path(__file__).parent.absolute() / 'thetas.json'
    with open(p, 'w') as f:
        f.write(dumps(data))
    print('SUCCESS')


def calculate_thetas(data: np.ndarray) -> None:
    km = list(data[1:, 0])
    price = list(data[1:, 1])
    t0, t1 = train_model(0.0, 0.0, km, price)
    write_to_file(t0, t1)


if __name__ == '__main__':
    p = Path(__file__).parent.absolute() / 'data.csv'
    if not p.exists():
        print(f'File {p} does not exist.')
        exit(1)
    calculate_thetas(np.genfromtxt(p, delimiter=','))
    exit(0)

