import numpy as np
import matplotlib.pyplot as plot
import pandas
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


def main():
    dataset = pandas.read_csv('data.csv')
    x = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, 1].values

    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size = 1/3, random_state = 0)

    regression_model = LinearRegression()
    # Fit the data(train the model)
    regression_model.fit(xTrain, yTrain)
    plot.scatter(xTrain, yTrain, color = 'red')
    plot.plot(xTrain, regression_model.predict(xTrain), color = 'blue')
    plot.title('Angle vs Area (Training set)')
    plot.xlabel('Angle')
    plot.ylabel('Area')
    plot.show()
    # print regression_model.predict([[16], [17], [18]])


if __name__ == "__main__":
    main()