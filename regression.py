import requests
import pandas
import scipy
import numpy
import sys


TRAIN_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/kubric-hiring/linreg_test.csv"


def predict_price(area) -> float:
    """
    This method must accept as input an array `area` (represents a list of areas sizes in sq feet) and must return the respective predicted prices (price per sq foot) using the linear regression model that you build.

    You can run this program from the command line using `python3 regression.py`.
    """
    response = requests.get(TRAIN_DATA_URL)
    # YOUR IMPLEMENTATION HERE
    data=response.text
    data=data.split(',')
    n=len(data)
    x=[]
    y=[]
    spl=267
    #print('len',n)
    for i in range(n):
        if(i>0 and i<266):
            x.append(float(data[i]))
        if(i==266):
            x.append(17770.0)
        if(i>266):
            y.append(float(data[i]))
    x=numpy.array(x)
    y=numpy.array(y)
    n = numpy.size(x)

    m_x, m_y = numpy.mean(x), numpy.mean(y)

    SS_xy = numpy.sum(y * x) - n * m_y * m_x
    SS_xx = numpy.sum(x * x) - n * m_x * m_x

    b_1 = SS_xy / SS_xx
    b_0 = m_y - b_1 * m_x
    y_pred = b_0 + b_1 * x
    yn=len(area)
    res=[]
    for i in range(yn):
        res.append(b_0 + b_1 * area[i])
    return res
    #print(data)



if __name__ == "__main__":
    # DO NOT CHANGE THE FOLLOWING CODE
    from data import validation_data
    areas = numpy.array(list(validation_data.keys()))
    prices = numpy.array(list(validation_data.values()))
    predicted_prices = predict_price(areas)
    rmse = numpy.sqrt(numpy.mean((predicted_prices - prices) ** 2))
    try:
        assert rmse < 170
    except AssertionError:
        print(f"Root mean squared error is too high - {rmse}. Expected it to be under 170")
        sys.exit(1)
    print(f"Success. RMSE = {rmse}")
