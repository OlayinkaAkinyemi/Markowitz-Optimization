import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

def drawdown(return_series: pd.Series, initial_investment):
    #return_series = (return_series/100)+1
    wealth = initial_investment*np.cumprod(1+return_series)
    cummax = wealth.cummax()
    drawdown = (wealth-cummax)/cummax
    return {"wealth":wealth,
    "cummax":cummax,
    "drawdown":drawdown}

"""def value_at_risk(return_series:pd.Series, confidence, column):
    confidence = confidence/100
    return_series = return_series.sort_values(by=[column])[column]
    obs = return_series.shape[0]
    obs_removed = (1-confidence)*obs
    value_at_risk=return_series.iloc[int(obs_removed)]
    beyond_value_at_risk = (return_series.iloc[:int(obs_removed)]).mean()
    #print(f"value_at_risk:{round(-value_at_risk,2)}%\nbeyond_value_at_risk: {round(-beyond_value_at_risk,2)}%")
    return round(-value_at_risk,2), round(-beyond_value_at_risk,3)
"""
def var_historic(returns, level):
    if isinstance(returns, pd.DataFrame):
        return returns.aggregate(var_historic, level=level)
    elif isinstance(returns, pd.Series):
        return -np.percentile(returns, level)
    else:
        raise TypeError("Expected returns to be a DataFrame or a Series")
        

def skewness(return_series: pd.Series):
    demeaned_return = return_series - return_series.mean()
    demeaned_return_cubed = demeaned_return**3
    sd = np.std(return_series,ddof=0)
    numerator = demeaned_return_cubed.mean()
    denominator = sd**3
    return numerator/denominator

def annualize_ret(return_series: pd.Series, periods_per_year):
    num_periods = return_series.shape[0]
    overall_ret = (1+return_series).prod()
    period_ret = (overall_ret)**(1/num_periods)
    annual_ret = (period_ret)**(periods_per_year)-1
    return annual_ret

def annualize_volatility(return_series: pd.Series, periods_per_year):
    std_period = np.std(return_series)
    std_annual = std_period*(periods_per_year)**(0.5)
    return std_annual

def sharpe_ratio(return_series: pd.Series, rfr_annual, periods_per_year):
    # risk free rate is on an annual basis, so you want to convert to a periodic basis
    # all input risks are assumed in the R rathan than 1+R format.conversion to (1+R) needs to be done in a given function
    rfr_per_period = (1+rfr_annual)**(1/periods_per_year)-1
    excess_ret = return_series-rfr_per_period
    annual_excess_ret = annualize_ret(excess_ret, periods_per_year)
    annual_std = annualize_volatility(return_series, periods_per_year)
    sharpe = annual_excess_ret/annual_std
    return sharpe

#def portfolio_return(weights, return_series, periods_per_year):
#    return weights.T @ annualize_ret(return_series,12)

#def portfolio_volatility(weights, return_series):
#    covariance = return_series.cov()
#    return (weights.T @ covariance @ weights)**(0.5)

def portfolio_return(weights, er):
    return weights@er

def portfolio_volatility(weights, cov):
    #covariance = return_series.cov()
    return (weights.T @ cov @ weights)**(0.5)

def minimize_vol(target_return, er,cov):
    n=er.shape[0]
    init_guess=np.repeat(1/n,n)
    bounds = ((0,1),)*n
    weight_constraint = {"type":"eq",
                        "fun": lambda weights: sum(weights)-1}
    target_constraint = {"type": "eq",
                        "fun": lambda weights, er: target_return - portfolio_return(weights, er),
                        "args": (er,),}
    weights = minimize(fm.portfolio_volatility, init_guess,  args=(cov,), method="SLSQP", 
                       options={"disp": False}, constraints=(weight_constraint, target_constraint),
                     bounds=bounds)
    return weights.x

def neg_sharpe_ratio(weights, rf, er, cov):
    portfolio_vola = portfolio_volatility(weights, cov)
    sharpe = (weights@er - rf)/portfolio_vola
    return -sharpe

def maximize_sharpe(rf, er, cov):
    n = cov.shape[0]
    init_guess = np.repeat(1/n,n)
    weights_sum_to_1 = {"type":"eq",
                      "fun": lambda weights: sum(weights)-1}
    bounds = ((0,1),)*n
    weights = minimize(neg_sharpe_ratio, init_guess, bounds=bounds, constraints=(weights_sum_to_1),
                       method="SLSQP", args=(rf, er, cov), options={"disp":False})
    return weights.x

"""
def plot_ef2(return_series, n_points, periods_per_year):
    weights = np.array([[w,1-w] for w in np.linspace(0,1,n_points)])
    #port_returns = portfolio_return(weights, return_series, periods_per_year).reshape(2,1)
    port_returns = [portfolio_return(weight,return_series,periods_per_year) for weight in weights]
    port_vola = [portfolio_volatility(weight, return_series) for weight in weights]
    dict_ = {"port_returns":port_returns, "port_vola":port_vola}
    df = pd.DataFrame(dict_)
    df.plot(x="port_vola", y="port_returns", style = ".-")
"""

def plot_ef(er, cov, rf, equal_weights=False, cml=False):
    n = er.shape[0]
    er_min = min(er)
    er_max = max(er)
    range_ = np.linspace(er_min, er_max, 100)
    portfolio_ret = [minimize_vol(i, er, cov)@er for i in range_]
    portfolio_vola = [portfolio_volatility(minimize_vol(i, er, cov), cov) for i in range_]
    fig, ax = plt.subplots()
    ax.scatter(portfolio_vola, portfolio_ret, s=5)
    ax.plot(portfolio_vola, portfolio_ret, label=f"{n}-asset frontier")
    ax.set_xlabel("risk"); ax.set_ylabel("return")
    
    if equal_weights:
        weights = np.repeat(1/n,n)
        portfolio_ret = weights@er
        portfolio_vola = portfolio_volatility(weights, cov)
        ax.scatter(portfolio_vola, portfolio_ret, s=20, label="equal weights portfolio")
    
    if cml:
        weights = maximize_sharpe(rf, er, cov)
        max_sharpe_ratio_return = weights@er
        max_sharpe_ratio_vola = portfolio_volatility(weights, cov)
        cml_x = [0, max_sharpe_ratio_vola]
        cml_y = [rf, max_sharpe_ratio_return]
        ax.scatter(cml_x, cml_y)
        ax.plot(cml_x, cml_y, "--", label =  "capital market line")
    
    ax.legend()
    ax.set_xlim(0); ax.set_ylim(0)
    plt.show()
    
    











