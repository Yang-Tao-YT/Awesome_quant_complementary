# %%
import pandas as pd
from arch.unitroot import ADF
import numpy as np
import re
import matplotlib.pyplot as plt
import statsmodels.api as sm


plt.rcParams['font.sans-serif']=['Arial Unicode MS']
plt.rcParams['axes.unicode_minus']=False

def TradeSig(prcLevel):
    n=len(prcLevel)
    signal=np.zeros_like(prcLevel)
    for i in range(1,n):
        if prcLevel[i-1]==1 and prcLevel[i]==2:
            signal[i]=-2
        elif prcLevel[i-1]==1 and prcLevel[i]==0:
            signal[i]=2
        elif prcLevel[i-1]==2 and prcLevel[i]==3:
            signal[i]=3
        elif prcLevel[i-1]==-1 and prcLevel[i]==-2:
            signal[i]=1
        elif prcLevel[i-1]==-1 and prcLevel[i]==0:
            signal[i]=-1
        elif prcLevel[i-1]==-2 and prcLevel[i]==-3:
            signal[i]=-3
    return(signal)

def test_adf(df):
    # 平稳序列
    result = ADF(df)
    return result.summary()

def SSD(pari_price):
    # 最小距离法
    priceX, priceY = pari_price.iloc[:,0] , pari_price.iloc[:,1]
    #return
    returnX=(priceX-priceX.shift(1))/priceX.shift(1).dropna()
    returnY=(priceY-priceY.shift(1))/priceY.shift(1).dropna()
    #price start with 1
    standardX=(returnX+1).cumprod()
    standardY=(returnY+1).cumprod()
    #ssd
    SSD=np.sum((standardX-standardY)**2)
    return(SSD)

def regression_test(Y,X):
    const_X = sm.add_constant(X)
    reg_results = sm.OLS(Y,const_X).fit()
    alpha = reg_results.params[0]
    beta = reg_results.params[1]
    spread = Y - beta * X - alpha
    sp_results = ADF(spread , trend='nc')
    return reg_results , sp_results


def TradeSim(priceX, priceY, position):
    n = len(position)
    size = 1000
    shareY = size * position
    shareX = [(-beta) * shareY[0] * priceY[0] / priceX[0]]
    print(shareY)
    cash = [2000]
    for i in range(1, n):
        shareX.append(shareX[i - 1])
        cash.append(cash[i - 1])
        if position[i - 1] == 0 and position[i] == 1:
            shareX[i] = (-beta) * shareY[i] * priceY[i] / priceX[i]
            cash[i] = cash[i - 1] - (shareY[i] * priceY[i] + shareX[i] * priceX[i])
        elif position[i - 1] == 0 and position[i] == -1:
            shareX[i] = (-beta) * shareY[i] * priceY[i] / priceX[i]
            cash[i] = cash[i - 1] - (shareY[i] * priceY[i] + shareX[i] * priceX[i])
        elif position[i - 1] == 1 and position[i] == 0:
            shareX[i] = 0
            cash[i] = cash[i - 1] + (shareY[i - 1] * priceY[i] + shareX[i - 1] * priceX[i])
        elif position[i - 1] == -1 and position[i] == 0:
            shareX[i] = 0
            cash[i] = cash[i - 1] + (shareY[i - 1] * priceY[i] + shareX[i - 1] * priceX[i])
    cash = pd.Series(cash, index=position.index)
    shareY = pd.Series(shareY, index=position.index)
    shareX = pd.Series(shareX, index=position.index)
    asset = cash + shareY * priceY + shareX * priceX
    account = pd.DataFrame({'Position': position, 'ShareY': shareY, 'ShareX': shareX, 'Cash': cash, 'Asset': asset})
    return (account)
# %%

if __name__ =='__main__':
    # %%
    # test the distance between 2 stock and pick the pair with smallest distance
    hist_data=pd.read_csv('sh50p.csv',index_col='Trddt')
    hist_data.index=pd.to_datetime(hist_data.index)

    price_BOC = hist_data['601988']['2014-01-01':'2015-01-01']
    price_pufa = hist_data['600000']['2014-01-01':'2015-01-01']

    pari_price = pd.concat([price_pufa, price_BOC] , 1) ; print(pari_price.shape)
    pari_price.plot()
    plt.show()
    distance = SSD(pari_price)
    print(distance)
    # %%
    # 平稳序列检测 # 原假设为非平稳序列
    # 中国银行log return and diff of log return
    log_price_BOC = np.log(price_BOC) ;
    log_return_BOC = log_price_BOC.diff().dropna()
    print(test_adf(log_price_BOC))
    print(test_adf(log_return_BOC))
    # 浦发银行log return and diff of log return
    log_price_pufa = np.log(price_pufa)
    log_return_pufa =  log_price_pufa.diff().dropna()
    print(test_adf(log_price_pufa))
    print(test_adf(log_return_pufa))
    # %%
    # plotting return
    log_price_BOC.plot(label='601988', style='--')
    log_price_pufa.plot(label='600000', style='-')
    plt.legend(loc='upper left')
    plt.title('中国银行与浦发银行的对数价格时序图')
    plt.show()
    #
    log_return_BOC.plot(label='601988', style='--')
    log_return_pufa.plot(label='600000', style='-')
    plt.legend(loc='lower left')
    plt.title('中国银行与浦发银行对数价格差分(收益率)')
    plt.show()
    # %% regression and errors test
    reg_results , sp_results = regression_test(log_price_BOC , log_price_pufa )
    print(reg_results.summary() , sp_results.summary())

    #%%
    '''trading'''
    '''paring period'''
    # 平稳序列检测 # 原假设为非平稳序列
    price_BOC = hist_data['601988']['2014-01-01':'2015-01-01']
    price_pufa = hist_data['600000']['2014-01-01':'2015-01-01']
    df = pd.concat([price_BOC , price_pufa] , 1)
    log_price_BOC = np.log(price_BOC)
    log_price_pufa = np.log(price_pufa)
    print(test_adf(log_price_BOC) , test_adf(log_price_pufa))
    log_return_BOC = log_price_BOC.diff().dropna()
    log_return_pufa =  log_price_pufa.diff().dropna()
    print(test_adf(log_return_BOC) , test_adf(log_return_pufa))
    #%%
    # #协整关系检验 残值平稳检验
    reg_results , sp_results = regression_test(log_price_pufa, log_price_BOC)
    print(reg_results.summary()  , sp_results.summary())
    alpha = reg_results.params[0]
    beta = reg_results.params[1]
    Spread_reg= log_price_pufa - beta * log_price_BOC - alpha
    mu = np.mean(Spread_reg)
    sd = np.std(Spread_reg)
    Spread_reg.plot()
    plt.title('交易期价差序列(协整配对)')
    plt.axhline(y=mu, color='black')
    plt.axhline(y=mu + 0.2 * sd, color='blue', ls='-', lw=2)
    plt.axhline(y=mu - 0.2 * sd, color='blue', ls='-', lw=2)
    plt.axhline(y=mu + 1.5 * sd, color='green', ls='--', lw=2.5)
    plt.axhline(y=mu - 1.5 * sd, color='green', ls='--', lw=2.5)
    plt.axhline(y=mu + 2.5 * sd, color='red', ls='-.', lw=3)
    plt.axhline(y=mu - 2.5 * sd, color='red', ls='-.', lw=3)
    plt.show()
    #%%
    # trading period
    price_BOC = hist_data['601988'].loc['2015-01-01':'2015-06-30']
    price_pufa = hist_data['600000'].loc['2015-01-01':'2015-06-30']
    log_price_BOC = np.log(price_BOC)
    log_price_pufa = np.log(price_pufa)
    SpreadT = log_price_pufa - beta * log_price_BOC - alpha
    SpreadT.describe()

    SpreadT.plot()
    plt.title('交易期价差序列(协整配对)')
    plt.axhline(y=mu, color='black')
    plt.axhline(y=mu + 0.2 * sd, color='blue', ls='-', lw=2)
    plt.axhline(y=mu - 0.2 * sd, color='blue', ls='-', lw=2)
    plt.axhline(y=mu + 1.5 * sd, color='green', ls='--', lw=2.5)
    plt.axhline(y=mu - 1.5 * sd, color='green', ls='--', lw=2.5)
    plt.axhline(y=mu + 2.5 * sd, color='red', ls='-.', lw=3)
    plt.axhline(y=mu - 2.5 * sd, color='red', ls='-.', lw=3)
    plt.show()
    #%%
    level = (float('-inf'), mu - 2.5 * sd, mu - 1.5 * sd, mu - 0.2 * sd, mu + 0.2 * sd, mu + 1.5 * sd, mu + 2.5 * sd,
             float('inf'))
    prcLevel = pd.cut(SpreadT, level, labels=False) -3
    signal = TradeSig(prcLevel)
    position = [signal[0]]

    for i in range(1, len(signal)):
        position.append(position[-1])
        if signal[i] == 1:
            position[i] = 1
        elif signal[i] == -2:
            position[i] = -1
        elif signal[i] == -1 and position[i - 1] == 1:
            position[i] = 0
        elif signal[i] == 2 and position[i - 1] == -1:
            position[i] = 0
        elif signal[i] == 3:
            position[i] = 0
        elif signal[i] == -3:
            position[i] = 0
    #%%
    position = pd.Series(position, index=SpreadT.index)

    account1 = TradeSim(price_BOC, price_pufa, position)
    account1.tail()
    account1.loc[:,'Asset']
    account1.iloc[:, [1, 3, 4]].plot(style=['--', '-', ':'])
    plt.show()
    plt.title('配对交易账户')
    account1.iloc[:, [0, 3, 4]]