import pandas as pd
import statsmodels.api as sm

# read stocks return
stock=pd.read_table('stock.txt',sep='\t',index_col='Trddt')

HXBank = stock[stock.Stkcd== 600015]
HXBank.index=pd.to_datetime(HXBank.index)
HXRet=HXBank.loc[:,'Dretwd'].to_frame()

# read factors
ThreeFactors=pd.read_table('ThreeFactors.txt',sep='\t',
                           index_col='TradingDate')
ThreeFactors.index = pd.to_datetime(ThreeFactors.index)
temp_threefac = ThreeFactors.iloc[:,[2,4,6]]
temp_threefac = temp_threefac.loc['2014-01-02':,:]
temp_1 = pd.concat([HXRet ,temp_threefac ] , 1).dropna()

# regression
print(temp_1)
X = sm.add_constant(temp_1.iloc[:,1:])
Y = temp_1.iloc[:,0]
regression = sm.OLS(Y,X)
result = regression.fit()
result.summary()
const,riskpremium2,SMB2 , HML2 = result.params