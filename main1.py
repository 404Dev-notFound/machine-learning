# #video number 10 https://www.youtube.com/watch?v=pfmJtWEMEpo&list=PLu0W_9lII9ai6fAMHp-acBmJONT7Y4BSG&index=10
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn import datasets, linear_model
# from sklearn.metrics import mean_squared_error
#
# diabetes= datasets.load_diabetes()
#
# # (['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
# # print(diabetes.keys())
# # print(diabetes.data)
# # print(diabetes.DESCR)
#
# diabetes_X = diabetes.data[:,np.newaxis,2]
#
# # print(diabetes_X)
#
# diabetes_X_train =diabetes_X[:-30]
# diabetes_X_test = diabetes_X[-30:]
#
# diabetes_y_train = diabetes.target[:-30]
# diabetes_y_test = diabetes.target[-30:]
#
# model = linear_model.LinearRegression()
#
# model.fit(diabetes_X_train,diabetes_y_train)
#
# diabetes_y_predicted=model.predict(diabetes_X_test)
#
# print("mean squared error is : ", mean_squared_error(diabetes_y_test,diabetes_y_predicted))
#
# print("weights:",model.coef_)
# print("intercept:",model.intercept_)
#
# plt.scatter(diabetes_X_test,diabetes_y_test)
# plt.plot(diabetes_X_test,diabetes_y_predicted)
#
# plt.show()
#
#
#
#









import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

diabetes= datasets.load_diabetes()

# (['data', 'target', 'frame', 'DESCR', 'feature_names', 'data_filename', 'target_filename', 'data_module'])
# print(diabetes.keys())
# print(diabetes.data)
# print(diabetes.DESCR)

diabetes_X = np.array([[1],[2],[3]])

# print(diabetes_X)

diabetes_X_train =diabetes_X
diabetes_X_test = diabetes_X

diabetes_y_train = np.array([3,2,4])
diabetes_y_test = np.array([3,2,4])

model = linear_model.LinearRegression()

model.fit(diabetes_X_train,diabetes_y_train)

diabetes_y_predicted=model.predict(diabetes_X_test)

print("mean squared error is : ", mean_squared_error(diabetes_y_test,diabetes_y_predicted))

print("weights:",model.coef_)
print("intercept:",model.intercept_)

plt.scatter(diabetes_X_test,diabetes_y_test)
plt.plot(diabetes_X_test,diabetes_y_predicted)

plt.show()




