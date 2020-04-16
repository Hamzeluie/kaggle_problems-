from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from preprocess import train_data
from settings import *  # contain data path


y_data = train_data.Survived
x_data = train_data[[i for i in train_data.keys() if i != 'Survived']]

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.25)

model = RandomForestClassifier(n_estimators=30)
model2 = SVC(gamma='auto').fit(x_train, y_train)

model.fit(x_train, y_train)
print(f'train accuracy: {model.score(x_train, y_train)}')
print(f'test accuracy: {model.score(x_test, y_test)}')
