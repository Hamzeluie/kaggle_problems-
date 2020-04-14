import matplotlib.pyplot as plt
import seaborn as sns
from settings import *
import pandas as pd
import shap  # package used to calculate Shap values
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.metrics import log_loss, accuracy_score
import seaborn as sns


# ['PassengerId', 'Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp',
#        'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked'],



# pd.DataFrame({'PassengerId': , 'Survived': model.predict(x_data)}, columns=['Survived'])
train_data = pd.read_csv(TRAIN_PATH)
test_data = pd.read_csv(TEST_PATH)






ffeatures = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp',
            'Parch', 'Embarked']
features = ['Pclass', 'Sex', 'Age', 'SibSp',
            'Parch',  'Embarked']
# features = ['Pclass', 'Sex', 'Age', 'Fare']
ttest_data = test_data[ffeatures]
test_data = test_data[features]

y_data = train_data.Survived
x_data = train_data[features]
x_data['Embarked'] = x_data['Embarked'].fillna('S')

test_data['Embarked'] = test_data['Embarked'].fillna('S')

embarked_dict = {'C': 0, 'Q': 1, 'S': 2}
embarked = [embarked_dict[i] for i in x_data['Embarked']]
test_embarked = [embarked_dict[i] for i in test_data['Embarked']]

x_data['Embarked'] = embarked
test_data['Embarked'] = test_embarked

sex_dict = {'female': 0, 'male': 1}
sex = [sex_dict[i] for i in x_data['Sex']]
test_sex = [sex_dict[i] for i in test_data['Sex']]

x_data['Sex'] = sex
test_data['Sex'] = test_sex

x_data['Age'] = x_data['Age'].fillna(29.)
test_data['Age'] = test_data['Age'].fillna(29.)

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.25)


model = RandomForestClassifier(n_estimators=30)


model.fit(x_train, y_train)

pred = model.predict(x_test)
score = log_loss(y_test, pred)
print(accuracy_score(y_test, pred))


explainer = shap.TreeExplainer(model)

# calculate shap values. This is what we will plot.
# Calculate shap_values for all of val_X rather than a single row, to have more data for plot.
shap_values = explainer.shap_values(x_test)

# Make plot. Index of [1] is explained in text below.
shap.summary_plot(shap_values[1], x_test)