#! /usr/bin/env python3
from json import loads
from pathlib import Path
from subprocess import Popen, PIPE
from typing import Tuple


def get_thetas() -> Tuple[float, float]:
    p = Path(__file__).parent.absolute() / 'thetas.json'

    if not p.exists():
        while True:
            run_script = input("The model has not been trained yet. Would you "
                               "like to run the training script (y/n)?: ").lower()
            if run_script in ['y', 'yes']:
                p_tr = Path(__file__).parent.absolute() / 'training.py'
                if not p_tr.exists():
                    print('Script training.py is absent!')
                    exit(1)
                with Popen(p_tr, stdout=PIPE) as proc:
                    out = str(proc.stdout.read())
                    if 'SUCCESS' not in out:
                        print(out)
                        exit(0)
                    break
            elif run_script in ['n', 'no']:
                print('Run the script training.py yourself to train the model.')
                exit(0)
            print('Unknown option, try again.')

    with open(p) as f:
        data = loads(f.read())
        theta0 = data.get('theta0', None)
        theta1 = data.get('theta1', None)
        if not any([theta0, theta1]):
            print(f"Some of data is absent: theta0 = {theta0}, theta1 = {theta1}"
                  "rerun the training script")
            exit(1)
        return theta0, theta1


def predict_price(mileage: int) -> None:
    theta0 = theta1 = 0
    theta0, theta1 = get_thetas()
    price = theta0 + (theta1 * mileage)
    print(f"Estimated price for mileage {mileage} is {price}.")
    exit(0)


if __name__ == '__main__':
    while True:
        mileage = input("Enter mileage for the car: ")
        try:
            mileage = int(mileage)
            predict_price(mileage)
        except ValueError:
            print("Value must be int, try again")
        except TypeError:
            print("Theta values must be numbers")
            exit(1)