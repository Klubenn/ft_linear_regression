#! /usr/bin/env python3

from json import dumps
from pathlib import Path
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt

# This value can be changed to make calculations faster, but should not exceed 1.7
LEARNING_RATE = 0.1

# This value can be changed to make calculations more precise
ERROR = 0.0000001


def estimated_price(t0, t1, km):
    return t1 * km + t0


def calculate_square_error(t0, t1, km, price) -> float:
    sum_ = sum([(estimated_price(t0, t1, x) - y) ** 2 for x, y in zip(km, price)])
    return sum_ / (2 * len(km))


def get_tmp_t0(t0, t1, km, price):
    size = len(km)
    sum_ = sum([estimated_price(t0, t1, km_i) - price_i for km_i, price_i in zip(km, price)])
    return (LEARNING_RATE / size) * sum_


def get_tmp_t1(t0, t1, km, price):
    size = len(km)
    sum_ = sum([(estimated_price(t0, t1, km_i) - price_i) * km_i for km_i, price_i in zip(km, price)])
    return (LEARNING_RATE / size) * sum_


def train_model(t0: float, t1: float, km: np.ndarray, price: np.ndarray) -> Tuple[float, float]:
    sq_er = calculate_square_error(t0, t1, km, price)
    i = 0
    while True:
        i += 1
        tmp_t0 = t0 - get_tmp_t0(t0, t1, km, price)
        tmp_t1 = t1 - get_tmp_t1(t0, t1, km, price)
        t0 = tmp_t0
        t1 = tmp_t1
        tmp_sq_er = calculate_square_error(t0, t1, km, price)
        if abs(tmp_sq_er - sq_er) < ERROR:
            break
        sq_er = tmp_sq_er
    return t0, t1


def write_to_file(t0, t1) -> None:
    data = {'theta0': t0, 'theta1': t1}
    p = Path(__file__).parent.absolute() / 'thetas.json'
    with open(p, 'w') as f:
        f.write(dumps(data))
    print('SUCCESS')


def normalize_data(data: np.ndarray):
    return (data - min(data)) / (max(data) - min(data))


def denormalize_thetas_negative(x, y):
    t0 = (max(y) * max(x) - min(y) * min(x)) / (max(x) - min(x))
    t1 = (min(y) - t0) / max(x)
    return t0, t1


def denormalize_thetas_positive(x, y):
    t0 = (max(y) * min(x) - min(y) * max(x)) / (min(x) - max(x))
    t1 = (min(y) - t0) / min(x)
    return t0, t1


def calculate_thetas(data: np.ndarray) -> None:
    km = data[1:, 0]
    price = data[1:, 1]
    km_n = normalize_data(km)
    t0, t1 = train_model(0.0, 0.0, km_n, price)
    y_min = estimated_price(t0, t1, 1)
    y_max = estimated_price(t0, t1, 0)
    t0, t1 = denormalize_thetas_negative(km, [y_min, y_max]) if t1 < 0 \
        else denormalize_thetas_positive(km, [y_min, y_max])
    write_to_file(t0, t1)


if __name__ == '__main__':
    p = Path(__file__).parent.absolute() / 'data.csv'
    if not p.exists():
        print(f'File {p} does not exist.')
        exit(1)
    calculate_thetas(np.genfromtxt(p, delimiter=','))
    exit(0)

