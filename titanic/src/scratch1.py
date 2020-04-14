from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.svm import SVC
from sklearn.inspection import permutation_importance
from sklearn.metrics import log_loss
from sklearn.model_selection import train_test_split
import pandas as pd
import shap
from settings import *  # contain data path


features = ['Pclass', 'Sex', 'Age', 'SibSp',
            'Parch', 'Embarked']

ffeatures = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp',
            'Parch', 'Embarked']

"""data features understanding"""

train_data = pd.read_csv(TRAIN_PATH)
print(train_data.describe())

y_data = train_data.Survived
x_data = train_data[features]
x_data['Embarked'] = x_data['Embarked'].fillna('S')

embarked_dict = {'C': 0, 'Q': 1, 'S': 2}
embarked = [embarked_dict[i] for i in x_data['Embarked']]
x_data['Embarked'] = embarked
sex_dict = {'female': 0, 'male': 1}
sex = [sex_dict[i] for i in x_data['Sex']]
x_data['Sex'] = sex
x_data['Age'] = x_data['Age'].fillna(29.)
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=.25)

model = RandomForestClassifier(n_estimators=30)
model2 = SVC(gamma='auto').fit(x_train, y_train)

model.fit(x_train, y_train)

pred = model.predict(x_test)
pred2 = model2.predict(x_test)
explainer = shap.TreeExplainer(model, x_train)
shap_values = explainer.shap_values(x_test)
shap.summary_plot(shap_values[1])
num_transformer = SimpleImputer(strategy='most_frequent')
one_hot = OneHotEncoder(handle_unknown='ignore')
cat_transformer = Pipeline(steps=[
    (
        'imputer', SimpleImputer(strategy='most_frequent')
    ),
    (
        'one_hot', OneHotEncoder(handle_unknown='ignore')
    )
])
transformer = ColumnTransformer(transformers=[
    ('num', num_transformer, ['Age', 'Fare']),
    ('cat', cat_transformer, ['Pclass', 'SibSp', 'Parch', 'Sex', 'Embarked'])
])
model = RandomForestRegressor(n_estimators=50)
my_pipeline = Pipeline(steps=[
    ('preprocess', transformer)
    , ('model', model)
])
my_pipeline.fit(x_train, y_train)
pred = my_pipeline.predict(x_test)
score = log_loss(y_test, pred)

# [i for i in x_train.columns if x_train[i].isnull().any()]
x_train = pd.DataFrame(num_transformer.fit_transform(x_train), columns=features)
one_hot.fit_transform(x_train)
explainer = shap.TreeExplainer(model, x_train)

shap_values = explainer.shap_values(x_test)

shap.summary_plot(shap_values[1], x_test)
result = permutation_importance(my_pipeline, x_test, y_test, n_repeats=10, random_state=42, n_jobs=2)
sorted_idx = result.importances_mean.argsort()
