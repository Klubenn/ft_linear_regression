#! /usr/bin/env python3

from json import dumps
from pathlib import Path
from typing import Union

import numpy as np
import matplotlib.pyplot as plt


class MyModel:
    # This value can be changed to make calculations faster, but should not exceed 1.7
    LEARNING_RATE = 0.1

    # This value can be changed to make calculations more precise
    ERROR = 0.0000001

    def __init__(self, data):
        self.km = data[1:, 0]
        self.price = data[1:, 1]
        self.km_norm = self.normalize_data(self.km)
        self.t0 = 0.0
        self.t1 = 0.0
        self.sq_err = 0.0

    @staticmethod
    def normalize_data(array: np.array):
        """
        Normalizes data by converting the minimum value to 0 and maximum value
        to 1 shifting the values in between correspondingly.
        """
        return (array - min(array)) / (max(array) - min(array))

    def save_thetas(self):
        """
        Saves thetas to the file in json format
        """
        data = {'theta0': self.t0, 'theta1': self.t1}
        p = Path(__file__).parent.absolute() / 'thetas.json'
        with open(p, 'w') as f:
            f.write(dumps(data))
        print('SUCCESS')

    def denormalize_thetas(self):
        """
        Converts thetas calculated for the normalized dataset to the initial
        one depending on the positive or negative value of theta1
        """
        y_min = self.estimated_price(1)
        y_max = self.estimated_price(0)
        if self.t1 < 0:
            self.t0 = (y_max * max(self.km) - y_min * min(self.km)) / \
                      (max(self.km) - min(self.km))
            self.t1 = (y_min - self.t0) / max(self.km)
        else:
            self.t0 = (y_max * min(self.km) - y_min * max(self.km)) / \
                      (min(self.km) - max(self.km))
            self.t1 = (y_min - self.t0) / min(self.km)

    def estimated_price(self, x: Union[int, float]):
        """
        Calculates the price for the given value on the x-axis depending on
        the current value of thetas
        """
        return self.t1 * x + self.t0

    def calculate_square_error(self) -> float:
        """
        Calculates square error depending on the current value of thetas
        """
        summ = sum([(self.estimated_price(km_i) - price_i) ** 2
                    for km_i, price_i in zip(self.km_norm, self.price)])
        return summ / (2 * len(self.km_norm))

    def _calculate_t0(self):
        """
        Calculates t0 depending on the current values of t0 and learning rate
        """
        size = len(self.km)
        summ = sum([self.estimated_price(km_i) - price_i
                    for km_i, price_i in zip(self.km_norm, self.price)])
        return (self.LEARNING_RATE / size) * summ

    def _calculate_t1(self):
        """
        Calculates t1 depending on the current values of t0 and learning rate
        """
        size = len(self.km)
        summ = sum([(self.estimated_price(km_i) - price_i) * km_i
                    for km_i, price_i in zip(self.km_norm, self.price)])
        return (self.LEARNING_RATE / size) * summ

    def train_model(self) -> None:
        """
        Calculates thetas over and over again till the difference between the
        two consecutive calculations' square error does not exceed the
        predefined value
        """
        while True:
            tmp_t0 = self.t0 - self._calculate_t0()
            tmp_t1 = self.t1 - self._calculate_t1()
            self.t0 = tmp_t0
            self.t1 = tmp_t1
            tmp_sq_er = self.calculate_square_error()
            if abs(tmp_sq_er - self.sq_err) < self.ERROR:
                break
            self.sq_err = tmp_sq_er


def calculate_thetas(data: np.ndarray) -> None:
    """
    Creates an instance of the model, trains it and saves the result.
    """
    model = MyModel(data)
    model.train_model()
    model.denormalize_thetas()
    model.save_thetas()


def main() -> None:
    """
    Opens the dataset and initiates calculations
    """
    p = Path(__file__).parent.absolute() / 'data.csv'
    if not p.exists():
        print(f'File {p} does not exist.')
        exit(1)
    calculate_thetas(np.genfromtxt(p, delimiter=','))


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
