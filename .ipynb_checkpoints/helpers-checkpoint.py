{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import matplotlib.mlab as mlab\n",
    "\n",
    "import statsmodels.api as sm\n",
    "import statsmodels.tsa.api as smt\n",
    "from statsmodels.tsa.arima_model import ARIMA\n",
    "import statsmodels.tsa.stattools as ts\n",
    "from statsmodels.tsa.stattools import adfuller\n",
    "\n",
    "\n",
    "import warnings\n",
    "warnings.filterwarnings('ignore')\n",
    "\n",
    "import itertools\n",
    "\n",
    "from sklearn.linear_model import LinearRegression\n",
    "from sklearn.linear_model import RidgeCV\n",
    "\n",
    "from pandas.plotting import autocorrelation_plot\n",
    "\n",
    "import re\n",
    "\n",
    "import sys\n",
    "import os\n",
    "\n",
    "from functools import reduce"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [],
   "source": [
    "def get_fuller_test(series):\n",
    "    values = series.values\n",
    "    result = adfuller(values)\n",
    "    print('ADF Statistic: %f' % result[0])\n",
    "    print('p-value: %f' % result[1])\n",
    "    print('Critical Values:')\n",
    "    for key, value in result[4].items():\n",
    "        print('\\t%s: %.3f' % (key, value))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "metadata": {},
   "outputs": [],
   "source": [
    "def calc_RMSE(validation_data, prediction_data):\n",
    "   \"\"\"\n",
    "   Calculate RMSE\n",
    "   \"\"\"\n",
    "   a = np.array(validation_data)\n",
    "   b = np.array(prediction_data)\n",
    "\n",
    "   return np.sqrt(np.mean((b-a)**2))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "metadata": {},
   "outputs": [],
   "source": [
    "def make_plots(data, lags=None):\n",
    "    '''\n",
    "    plotting the data with specified number of lags.\n",
    "    plotting raw data, then ACF and PACF\n",
    "    '''\n",
    "    layout = (1, 3)\n",
    "    raw  = plt.subplot2grid(layout, (0, 0))\n",
    "    acf  = plt.subplot2grid(layout, (0, 1))\n",
    "    pacf = plt.subplot2grid(layout, (0, 2))\n",
    "    \n",
    "    data.plot(ax=raw, figsize=(12, 6))\n",
    "    smt.graphics.plot_acf(data, lags=lags, ax=acf)\n",
    "    smt.graphics.plot_pacf(data, lags=lags, ax=pacf)\n",
    "    sns.despine()\n",
    "    plt.tight_layout()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "metadata": {},
   "outputs": [],
   "source": [
    "def make_plots_2(data, lags=None):\n",
    "    '''\n",
    "    plotting rolling mean, rolling std and original as per number of lags\n",
    "    '''\n",
    "    rolling_mean = data.rolling(window = lags).mean()\n",
    "    rolling_std = data.rolling(window = lags).std()\n",
    "    \n",
    "    original = plt.plot(data, color='black',label = 'Original Timeseries')\n",
    "    mean = plt.plot(rolling_mean, color='red', label = 'Rolling Mean')\n",
    "    std = plt.plot(rolling_std, color='orange', label = 'Rolling Std')\n",
    "    plt.legend(loc='best')\n",
    "    plt.title('Original, Rolling Mean, Standard Deviation')\n",
    "    plt.show(block = False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "metadata": {},
   "outputs": [],
   "source": [
    "def find_best_params(array):\n",
    "    '''\n",
    "    findind the best parametes using grid lock search\n",
    "    '''\n",
    "    \n",
    "    # creating the  parameters as tuples\n",
    "    p = d = q = range(0, 3)\n",
    "    pdq = list(itertools.product(p, d, q))\n",
    "    results = {}\n",
    "    for param in pdq:\n",
    "        \n",
    "        try:\n",
    "            model = ARIMA(array ,order = param )\n",
    "            fit = model.fit()\n",
    "            results[str(param)] = fit.aic\n",
    "            \n",
    "        except:\n",
    "            continue\n",
    "\n",
    "    print(min(results, key = results.get))\n",
    "    return results"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "metadata": {},
   "outputs": [],
   "source": [
    "def foo():\n",
    "    print( \"Yodel\" )"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.1"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
