import pandas as pd
import numpy as np

from sklearn.datasets import fetch_california_housing

data = fetch_california_housing()

dtfr = pd.DataFrame(data=data.data, columns = data.feature_names) 

dtfr['Target'] = data.target

#데이터셋의 그래프들
#!pip install sweetviz 필요
import sweetviz as sv
report = sv.analyze(dtfr)
report.show_html("./report.html")


dtfr = dtfr.sample(axis = 0,frac = 1)

y = dtfr.iloc[:,-1].values

dtfr.drop(labels=['Target'], axis = 1, inplace=True)
x = dtfr.iloc[:,:].values


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.20, random_state = 42)

from sklearn.ensemble import RandomForestRegressor
model = RandomForestRegressor()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

from sklearn.metrics import r2_score
prepercentage = r2_score(y_test, y_pred) * 100
prepercentage = round(prepercentage,2)

"""
sample array
MedInc HousAge AveRooms Avebedrms Population AveOccup Latitude Longitude 
8.3252 41.0   6.984127  1.023810   322.0      2.55556  37.88    -122.23

        - MedInc        median income in block
        - HouseAge      median house age in block
        - AveRooms      average number of rooms
        - AveBedrms     average number of bedrooms
        - Population    block population
        - AveOccup      average house occupancy
        - Latitude      house block latitude
        - Longitude     house block longitude
"""

#input값 조정
inpt = np.array([8,41,7,1,322,2.5,37.8,-122])


inp1 = inpt.reshape((1,-1))

Houseprice = float(model.predict(inp1))
Houseprice = round(Houseprice,4)
print("\n"+str(Houseprice)+"(단위 : 100,000$)\n")
print("정확도 : "+str(prepercentage)+"%")
        