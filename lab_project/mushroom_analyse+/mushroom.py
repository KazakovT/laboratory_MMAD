import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.naive_bayes import BernoulliNB


# Загрузка и подготовка данных
mushroom = fetch_ucirepo(id=73)
X = mushroom.data.features
y = mushroom.data.targets
selected_features = [
    'stalk-surface-above-ring', 
    'stalk-surface-below-ring',
    'stalk-color-above-ring',
    'stalk-color-below-ring', 
    'veil-type'
]

X = X[selected_features]
df = pd.DataFrame(X, columns=selected_features)
df['target'] = y

# Анализ распределения признаков
for feature in selected_features:
    print(f"\nРаспределение для признака '{feature}':")
    print(df.groupby([feature, 'target']).size())

# Визуализация распределений
fig, axes = plt.subplots(2, 3, figsize=(15, 12))
axes = axes.flatten()
for idx, feature in enumerate(selected_features):
    ax = axes[idx]
    sns.countplot(x=feature, hue='target', data=df, ax=ax)
    ax.set_title(f'Распределение: {feature}')
    ax.tick_params(axis='x', rotation=45)
fig.savefig('Распределения.png')

# Кодирование категориальных признаков
label_encoder = LabelEncoder()
for column in X.columns:
    if X[column].dtype == object:
        X[column] = label_encoder.fit_transform(X[column])

# Разделение данных
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Подготовка DataFrame для анализа
df_train = pd.DataFrame(X_train, columns=X_train.columns)
df_train['target'] = y_train

df_test = pd.DataFrame(X_test, columns=X_test.columns)
df_test['target'] = y_test
selected_features=X_train.columns

# Построение решающих функций на основе частот
accuracies = {}

for feature in selected_features:
    # Расчет частот
    freq_train = df_train.groupby([feature, 'target']).size().reset_index(name='count')
    freq_test = df_test.groupby([feature, 'target']).size().reset_index(name='count')

    X_train_freq = freq_train.drop('target', axis=1).values
    y_train_freq = freq_train['target'].values
    X_test_freq = freq_test.drop('target', axis=1).values
    y_test_freq = freq_test['target'].values

    # Обучение модели
    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_freq, y_train_freq)

    # Оценка точности
    y_train_pred = model.predict(X_train_freq)
    y_test_pred = model.predict(X_test_freq)

    accuracy_train = accuracy_score(y_train_freq, y_train_pred)
    accuracy_test = accuracy_score(y_test_freq, y_test_pred)

    accuracies[feature] = {
        'accuracy_train': accuracy_train,
        'accuracy_test': accuracy_test
    }

for feature, acc in accuracies.items():
    print(f"\nПризнак: {feature}")
    print(f"  Точность на обучении: {acc['accuracy_train']}")
    print(f"  Точность на тесте: {acc['accuracy_test']}")

most_informative_feature = max(
    accuracies, 
    key=lambda x: accuracies[x]['accuracy_test']
)
print(f"\nНаиболее информативный признак: {most_informative_feature}")

# One-hot кодирование для наивного Байеса
X = mushroom.data.features
y = mushroom.data.targets
X=X[selected_features]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
one_hot_encoded_training_predictors = pd.get_dummies(X_train)
one_hot_encoded_test_predictors = pd.get_dummies(X_test)
X_train, X_test = one_hot_encoded_training_predictors.align(
    one_hot_encoded_test_predictors,
    join='left',
    axis=1
)
X_train = X_train.drop('stalk-color-below-ring_y', axis=1)
X_test = X_test.drop('stalk-color-below-ring_y', axis=1)

# Наивный Байес из sklearn
y_train = np.ravel(y_train)
y_test = np.ravel(y_test)

model = BernoulliNB()
model.fit(X_train, y_train)

y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

accuracy_train = accuracy_score(y_train, y_train_pred)
accuracy_test = accuracy_score(y_test, y_test_pred)


print(f"Точность на обучении: {accuracy_train}")
print(f"Точность на тесте: {accuracy_test}")


# Собственная реализация наивного Байеса
class NaiveBayesClassifier:
    def __init__(self):
        self.class_counts = {}
        self.class_probabilities = {}
        self.feature_counts = {}
        self.feature_probabilities = {}
        self.classes = []

    def fit(self, X, y):
        self.classes = np.unique(y)

        for class_ in self.classes:
            # Подсчет количества объектов класса
            self.class_counts[class_] = np.sum(y == class_)
            self.class_probabilities[class_] = self.class_counts[class_] / len(y)

            # Индексы объектов данного класса
            class_indices = np.where(y == class_)[0]
            class_features = X[class_indices]

            # Подсчет частот признаков для класса
            self.feature_counts[class_] = {}
            self.feature_probabilities[class_] = {}

            for feature in range(X.shape[1]):
                unique_values, counts = np.unique(
                    class_features[:, feature], 
                    return_counts=True
                )
                self.feature_counts[class_][feature] = dict(zip(unique_values, counts))
                self.feature_probabilities[class_][feature] = {}

                for value in unique_values:
                    self.feature_probabilities[class_][feature][value] = self.feature_counts[class_][feature][value] / self.class_counts[class_]

    def predict(self, X):
        predictions = []

        for instance in X:
            class_scores = {}

            for class_ in self.classes:
                # Логарифм априорной вероятности класса
                class_score = np.log(self.class_probabilities[class_])

                # Суммирование логарифмов вероятностей признаков
                for feature in range(X.shape[1]):
                    value = instance[feature]
                    feature_probability = self.feature_probabilities[class_][feature].get(value)

                    if feature_probability is not None:
                        class_score += np.log(feature_probability)

                class_scores[class_] = class_score

            # Выбор класса с максимальной вероятностью
            predicted_class = max(class_scores, key=class_scores.get)
            predictions.append(predicted_class)

        return predictions

naive_bayes = NaiveBayesClassifier()
naive_bayes.fit(X_train.to_numpy(), y_train)

y_pred = naive_bayes.predict(X_test.to_numpy())
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность: {accuracy}")

# Регуляризация сглаживанием Лапласа
alpha = 0.1  # Параметр сглаживания Лапласа

for feature in selected_features:
    freq_train = df_train.groupby([feature, 'target']).size().reset_index(name='count')
    freq_test = df_test.groupby([feature, 'target']).size().reset_index(name='count')

    # Добавление сглаживания Лапласа к частотам
    freq_train['count'] = freq_train['count'] + alpha
    freq_test['count'] = freq_test['count'] + alpha

    X_train_freq = freq_train.drop('target', axis=1).values
    y_train_freq = freq_train['target'].values
    X_test_freq = freq_test.drop('target', axis=1).values
    y_test_freq = freq_test['target'].values

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train_freq, y_train_freq)
    y_train_pred = model.predict(X_train_freq)
    y_test_pred = model.predict(X_test_freq)

    accuracy_train = accuracy_score(y_train_freq, y_train_pred)
    accuracy_test = accuracy_score(y_test_freq, y_test_pred)

    accuracies[feature] = {
        'accuracy_train': accuracy_train,
        'accuracy_test': accuracy_test
    }

for feature, acc in accuracies.items():
    print(f"\nПризнак: {feature}")
    print(f"  Точность на обучении: {acc['accuracy_train']}")
    print(f"  Точность на тесте: {acc['accuracy_test']}")


most_informative_feature = max(
    accuracies,
      key=lambda x: accuracies[x]['accuracy_test']
)
print(f"\nНаиболее информативный признак (со сглаживанием): {most_informative_feature}")
plt.tight_layout()
plt.show()