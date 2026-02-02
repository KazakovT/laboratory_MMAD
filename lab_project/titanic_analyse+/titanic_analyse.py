import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn import tree
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier

# Настройка стиля графиков
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

# Список для хранения всех фигур
figures = []


#Загрузка данных
df = pd.read_csv('gender_submission.csv')
test = pd.read_csv('test.csv')
train = pd.read_csv('train.csv')

# Объединение данных
test = pd.concat([df.iloc[:, -1], test], axis=1)
t_data = pd.concat([train, test], ignore_index=True)

# Анализ данных
variable_stats = t_data.describe()
print(f'\nСтатистика по числовым переменным:\n{variable_stats}')

gender_count = t_data['Sex'].value_counts()
print(f'\nРаспределение по полу:\n{gender_count}')

num_objects = len(t_data)
print(f'\nКоличество объектов: {num_objects}')

missing_values = t_data.isnull().sum()
print('\nКоличество пропущенных значений:')
print(missing_values)
#Выбор и подготовка признаков
t_data = t_data[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']].dropna()
t_data['Sex'] = t_data['Sex'].map({'female': 0, 'male': 1})

X = t_data.drop('Survived', axis=1)
y = t_data['Survived']

#Построение и визуализация дерева решений
decision_tree = DecisionTreeClassifier(max_depth=3)
decision_tree.fit(X, y)

# Визуализация дерева решений
fig1, ax1 = plt.subplots(figsize=(14, 8))
tree.plot_tree(
    decision_tree,
    feature_names=X.columns, 
    class_names=['Not Survived', 'Survived'], 
    filled=True, 
    rounded=True,
    fontsize=10,
    ax=ax1
)
ax1.set_title('Дерево решений для предсказания выживаемости на Титанике', fontsize=14)
fig1.tight_layout()
fig1.savefig('Дерево_решений.png')
figures.append(('Дерево решений', fig1))

y_pred = decision_tree.predict(X)
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)

print(f'\nМетрики качества на всем наборе данных:')
print(f'Accuracy:  {accuracy}')
print(f'Precision: {precision}')
print(f'Recall:    {recall}')

# Извлечение правил дерева
def get_decision_rules(tree, feature_names, class_names):
    rules = [ ]

    def recurse(node, rule):
        if tree.tree_.feature[node] != -2:
            feature = feature_names[tree.tree_.feature[node]]
            threshold = tree.tree_.threshold[node]

            rule_left = rule + f" {feature} <= {threshold}, "
            recurse(tree.tree_.children_left[node], rule_left)

            rule_right = rule + f" {feature} > {threshold}, "
            recurse(tree.tree_.children_right[node], rule_right)
        else:
            class_idx = tree.tree_.value[node][0].argmax()
            class_name = class_names[class_idx]
            rule_str = rule.replace("Sex <= 0.5", "Sex = female").replace("Sex > 0.5", "Sex = male") + f" then {class_name}"
            rules.append(rule_str)

    recurse(0, '')

    return rules

decision_rules = get_decision_rules(
    decision_tree, 
    X.columns, 
    ['Not Survived', 'Survived']
    )

print('\nПравила дерева решений:')
for rule in decision_rules:
    print(rule)
#Подготовка данных для кросс-валидации
train = train[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']].dropna()
train['Sex'] = train['Sex'].map({'female': 0, 'male': 1})

test = test[['Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Survived']].dropna()
test['Sex'] = test['Sex'].map({'female': 0, 'male': 1})

X_train = train.drop('Survived', axis=1)
y_train = train['Survived']
X_test=test.drop('Survived', axis=1)
y_test=test['Survived']

train_scores = []
test_scores = []
max_depths = range(1, 21)

for depth in max_depths:
    model = DecisionTreeClassifier(max_depth=depth)

    model.fit(X_train, y_train)
    train_score =np.mean(cross_val_score(model, X_train, y_train, cv=5))
    train_scores.append(train_score)
    y_pred=model.predict(X_test)
    test_scores.append(accuracy_score(y_test, y_pred))


fig2, ax2 = plt.subplots(figsize=(10, 6))
ax2.plot(max_depths, train_scores, 'o-', label='Кросс-валидация (train)', linewidth=2, markersize=8)
ax2.plot(max_depths, test_scores, 's-', label='Тестовая выборка', linewidth=2, markersize=8)
ax2.set_xlabel('Максимальная глубина дерева', fontsize=12)
ax2.set_ylabel('Accuracy', fontsize=12)
ax2.set_title('Зависимость качества от глубины дерева', fontsize=14)
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.set_xticks(range(1, 21, 2))
fig2.tight_layout()
fig2.savefig('глубина_дерева.png')
figures.append(('Зависимость от глубины дерева', fig2))

# Сравнение критериев разбиения
crits=['entropy', 'gini']
for crit in crits:
    model = DecisionTreeClassifier(max_depth=3,criterion=crit)

    model.fit(X_train, y_train)
    train_score =np.mean(cross_val_score(model, X_train, y_train, cv=5))
    y_pred=model.predict(X_test)
    test_score=accuracy_score(y_test, y_pred)
    print(f"\nКритерий: {crit}")
    print(f"  CV Accuracy: {train_score}")
    print(f"  Test Accuracy: {test_score}")
    
#Градиентный бустинг

model = GradientBoostingClassifier(n_estimators=3, random_state=42)
model.fit(X, y)
num_trees = 3

for i in range(num_trees):
  fig_gb, ax_gb = plt.subplots(figsize=(12, 6))
  plot_tree(
      model.estimators_[i, 0], 
      feature_names=X.columns, 
      class_names=['Not Survived', 'Survived'], 
      filled=True, 
      rounded=True
  )
  ax_gb.set_title(f'Градиентный бустинг: Дерево {i+1}', fontsize=14)
  fig_gb.tight_layout()
  fig_gb.savefig(f'Градиентный бустинг - Дерево {i+1}.png')
  figures.append((f'Градиентный бустинг - Дерево {i+1}', fig_gb))

feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns, 
    'Importance': feature_importances
})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

print("\nЗначимость переменных в градиентном бустинге:")
print(importance_df)

train_scores = []
test_scores = []
max_trees = range(1, 21)

for n in max_trees:
    model = GradientBoostingClassifier(n_estimators=n, random_state=42)

    model.fit(X_train, y_train)
    train_score =np.mean(cross_val_score(model, X_train, y_train, cv=5))
    train_scores.append(train_score)
    y_pred=model.predict(X_test)
    test_scores.append(accuracy_score(y_test, y_pred))

fig_gb_curve, ax_gb_curve = plt.subplots(figsize=(10, 6))
ax_gb_curve.plot(max_depths, train_scores, 'o-', label='Кросс-валидация (train)', linewidth=2, markersize=8)
ax_gb_curve.plot(max_depths, test_scores, 's-', label='Тестовая выборка', linewidth=2, markersize=8)
ax_gb_curve.set_xlabel('Количество деревьев', fontsize=12)
ax_gb_curve.set_ylabel('Accuracy', fontsize=12)
ax_gb_curve.set_title('Градиентный бустинг: зависимость качества от числа деревьев', fontsize=14)
ax_gb_curve.grid(True, alpha=0.3)
ax_gb_curve.legend()
ax_gb_curve.set_xticks(range(1, 21, 2))
fig_gb_curve.tight_layout()
fig_gb_curve.savefig('графики_точности.png')
figures.append(('Градиентный бустинг - зависимость от числа деревьев', fig_gb_curve))

#Случайный лес

model = RandomForestClassifier(max_depth=3, random_state=42)
model.fit(X, y)

# Значимость переменных
feature_importances = model.feature_importances_
importance_df = pd.DataFrame({
    'Feature': X.columns, 
    'Importance': feature_importances
})
importance_df.sort_values(by='Importance', ascending=False, inplace=True)

print("\nЗначимость переменных в случайном лесу:")
print(importance_df)
train_scores = []
test_scores = []
max_trees = range(1, 21)

for n in max_trees:
    model = RandomForestClassifier(max_depth=3, n_estimators=n, random_state=42)

    model.fit(X_train, y_train)
    train_score =np.mean(cross_val_score(model, X_train, y_train, cv=5))
    train_scores.append(train_score)
    y_pred=model.predict(X_test)
    test_scores.append(accuracy_score(y_test, y_pred))

fig_rf_curve, ax_rf_curve = plt.subplots(figsize=(10, 6))
ax_rf_curve.plot(max_depths, train_scores, 'o-', label='Кросс-валидация (train)', linewidth=2, markersize=8)
ax_rf_curve.plot(max_depths, test_scores, 's-', label='Тестовая выборка', linewidth=2, markersize=8)
ax_rf_curve.set_xlabel('Количество деревьев', fontsize=12)
ax_rf_curve.set_ylabel('Accuracy', fontsize=12)
ax_rf_curve.set_title('Случайный лес: зависимость качества от числа деревьев', fontsize=14)
ax_rf_curve.grid(True, alpha=0.3)
ax_rf_curve.legend()
ax_rf_curve.set_xticks(range(1, 21, 2))
fig_rf_curve.tight_layout()
fig_rf_curve.savefig('случайный_лес.png')
figures.append(('Случайный лес - зависимость от числа деревьев', fig_rf_curve))

#Вывод всех графиков
plt.rcParams['figure.max_open_warning'] = 50
n_figures = len(figures)
n_cols = 2
n_rows = (n_figures + n_cols - 1) // n_cols

for i, (title, fig) in enumerate(figures, 1):
    try:
        # Создаем новое окно для каждого графика
        fig_manager = plt.figure(fig.number)
        fig_manager.canvas.manager.set_window_title(f"График {i}: {title}")
        plt.show(block=False)  # Не блокируем выполнение
        print(f"  {i}. {title}")
    except Exception as e:
        print(f"  Ошибка при отображении графика '{title}': {e}")
plt.show(block=True)