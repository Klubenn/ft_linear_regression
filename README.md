### ft_linear_regression

The aim of the project is to create a model, that will be trained with the help of the dataset to predict the car price according to its mileage.

We'll perform the following steps:
1. Read the data from csv file and normalize the array with mileage. Normalization of data is needed for the program not to fall into infinity when it is performing calculations. So the smallest mileage becomes 0 and the highest becomes 1.
2. Calculate indexes of the linear equation - theta0(t0) and theta1(t1): `price = t1 * km + t0`. This action is performed in a loop and each time the standard deviation is calculated. When the difference between consecutive calculations of standard deviation are less than some value (ERROR) the calculations stop.
3. Calculate the minimum and the maximum hypothetical price with these indexes for `km = 0` and `km = 1`. With these values calculate new thetas, creating the line from two points with maximum and minimum hypothetical price and maximum and minimum real mileage (not normalized).
4. Calculations for the previous section depend on whether t1 is positive or negative - this influences the coordinates of the points - `x_min, y_max; x_max, y_min` - for negative t1 and `x_min, y_min; x_max, y_max` - for positive t1.

`training.py` - trains the model

`prediction.py` - predicts the price for the given mileage