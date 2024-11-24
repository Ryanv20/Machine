from sklearn.linear_model import LinearRegression

#regression model
reg = LinearRegression()

print("R")




X = [[1],[2],[3],[4],[5],[6]]
Y = [1.2, 2.5, 4.5, 3, 5, 4.7]

#fit
reg.fit(X,Y)

#Predict
reg.predict(([5.5], [2.3]))

result = reg.predict(([1], [2.3]))

print(result)
