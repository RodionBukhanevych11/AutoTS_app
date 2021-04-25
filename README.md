# AutoTS_app
Multifunctional app to work with multivariate timeseries. Build on PyQT5.
### Capabilities
* Splitting dataset on cointegrated series
* Transformation into stationary time series
* Forecasting methods:
    * VAR
    * Boosting on decision trees 
    * LSTM ...(in progress)
* Cross validation
* Metrics
* Validation & n_step prediction plots
### Screenshots
![Main windows](https://github.com/RodionBukhanevych11/AutoTS_app/tree/main/images/Screenshot_1.png)
![Forecasting window & metrics](https://github.com/RodionBukhanevych11/AutoTS_app/tree/main/images/Screenshot_2.png)
![Plots of forecasting](https://github.com/RodionBukhanevych11/AutoTS_app/tree/main/images/Screenshot_3.png)
#### Run app
py main.py <filepath>

!!! Dataset  should not contain any columns except pure time series
DateTime features extracting would be added in future!