import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# A class with three parameters but you only specify two when calling the class because the array parameter
# is initiated in the methods inside the class. The methods in the class reads a csv file and creates an array from
# the specified column by specifying the column parameter when you call the class.


class Gauss:

    def __init__(self, csv, column, array=None):
        self.csv = csv
        self.column = column
        self.array = array

    # Reads data from a csv file from a column of choice and creates an array from it
    def read_csv(self):
        df = pd.read_csv(self.csv)
        self.array = np.array(df[self.column])
        return self.array

    # Creates a matrix from the array and preparing it for concatenation and gauss elimination by reshaping it and
    # adding integers of ones to the second element in each row so it fits the calculation of AT * A * x = AT * Y

    def reshape_x_matrix(self):
        self.x_array = np.array([self.array, np.ones(len(self.array))], dtype=float).T
        return self.x_array

    # Reshaping the vector into one element rows so the calculation of AT * A * x = AT * Y

    def reshape_y_vector(self):
        self.y_array = self.array.reshape(-1, 1)
        return self.y_array

# A class where all the calculations of transpose, concatenate and gauss elimination is executed


class Calculate:

    rows = None
    cols = None
    Ab = None
    k = None
    m = None
    A2 = np.zeros((2, 2))

    # Creates a normal-equation, transposing and concatenates the equations before gauss-elimination

    @classmethod
    def transpose_concatenate(cls, A, b):

        a_trans = A.T
        a1 = np.dot(a_trans, A)
        b1 = np.dot(a_trans, b)

        cls.Ab = np.concatenate((a1, b1), axis=1)
        cls.rows = np.shape(cls.Ab)[0]
        cls.cols = np.shape(cls.Ab)[1]

        return cls.Ab

    # Executing gauss elimination that calculates k * x + m = y for the best fit line and returns the value of k and the
    # value of m where the value of k is the slope and the m is the crossing of the y-axis.

    @classmethod
    def gaussian_elimination(cls):

        # Gauss elimination

        solution_vector = np.zeros(cls.cols - 1)
        for i in range(cls.cols - 1):
            for j in range(i + 1, cls.rows):
                cls.Ab[j, :] = -(cls.Ab[j, i] / cls.Ab[i, i]) * cls.Ab[i, :] + cls.Ab[j, :]

        # Backwards substitution
        for i in np.arange(cls.rows - 1, -1, -1):
            solution_vector[i] = (cls.Ab[i, -1] - np.dot(cls.Ab[i, 0:cls.cols - 1], solution_vector)) \
                                 / cls.Ab[i, i]

        cls.k = np.round(solution_vector[0], 3)
        cls.m = np.round(solution_vector[1], 3)

    # Show a graph over the x and y-axis of rawdata from both the csv files readings and drawing a best fit line
    # through the plots showing the best fit line

    @classmethod
    def plot(cls):
        xa = np.linspace(x.array.min(), x.array.max())
        yb = cls.k * xa + cls.m
        plt.title("Average temperature in the year 2022")
        plt.xlabel("Temperature")
        plt.ylabel("Month")
        plt.scatter(x.array, y.array)
        plt.plot(x.array, cls.k * x.array + cls.m, 'y', label=f"Best line fit\nK = {cls.k}\nm = {cls.m}")
        plt.plot(xa, yb, 'y')
        plt.legend()
        plt.show()


# Creating class-objects

x = Gauss("smhi1.csv", "temp")
x1 = x.read_csv()
x2 = x.reshape_x_matrix()
y = Gauss("smhi1.csv", "month_nr")
y1 = y.read_csv()
y2 = y.reshape_y_vector()
d = Calculate()
d1 = d.transpose_concatenate(x.x_array, y.y_array)
Calculate.gaussian_elimination()
Calculate.plot()

