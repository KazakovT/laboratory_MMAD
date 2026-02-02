import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.feature_selection import f_regression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import cross_val_score


# Загрузка данных и предобработка
df = pd.read_csv('laptop_price.csv', encoding='latin-1')
df["Ram"]=df["Ram"].str.replace("GB", '').astype("int32")
df["Weight"]=df["Weight"].str.replace("kg",'' ).astype("float32")

# Кодирование категориальных переменных
label_encoder = LabelEncoder()
for col in df.select_dtypes(include=['object']):
    df[col] = label_encoder.fit_transform(df[col])

#Проверка данных
missing_values = df.isnull().sum()
print(f'\nКоличество пропущенных значений:{missing_values}')

duplicated_values =df.duplicated().sum()
print(f'\nКоличество повторяющихся значений:{duplicated_values}')

numeric_cols = df.select_dtypes(include=[np.number]).columns
#Диаграммы с усами
fig_box, axes_box = plt.subplots(nrows=4, ncols=4, figsize=(16, 10))
axes_box = axes_box.flatten()

for idx, col in enumerate(numeric_cols):
    ax = axes_box[idx]
    sns.boxplot(y=df[col], ax=ax, color='skyblue')
    ax.set_title(col, fontsize=10)
    ax.set_ylabel('')

for idx in range(len(numeric_cols), len(axes_box)):
    axes_box[idx].set_visible(False)

plt.suptitle('Распределение числовых признаков (boxplot)', fontsize=14)
plt.tight_layout()
plt.savefig('boxplot.png')
plt.show()

#Тепловая карта
correlation_matrix = df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(
    correlation_matrix, 
    annot=True, 
    cmap='coolwarm', 
    fmt='.2f',
    center=0,
    square=True,
    cbar_kws={"shrink": 0.8}
)
plt.title('Тепловая карта', fontsize=14)
plt.tight_layout()
plt.savefig('тепловая_карта.png')
plt.show()

# Гистограммы распределений
fig, axs = plt.subplots(ncols=5, nrows=3, figsize=(20, 10))
axs = axs.flatten()
for idx, (col_name, col_data) in enumerate(df.items()):
    axes = axs[idx]
    sns.histplot(col_data, ax=axes, kde=True)
    axes.set_title(col_name, fontsize=10)
    axes.set_xlabel('')
    axes.set_ylabel('')

for idx in range(len(df.columns), len(axs)):
    axs[idx].set_visible(False)
plt.tight_layout()
plt.savefig('гистограмма.png')
plt.show()

#Линейная регрессия
X = df.drop(columns=['Price_euros'])
y = df['Price_euros']

model = LinearRegression()
model.fit(X, y)

f_values, p_values = f_regression(X, y)

results = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_,
    'F-Value': f_values,
    'P-Value': p_values
})


results['Absolute Coefficient'] = abs(results['Coefficient'])
results = results.sort_values('Absolute Coefficient', ascending=False)

print(f"\nРезультаты линейной регрессии:{results}")

#Градиентный бустинг
linear_regression = LinearRegression()

linear_regression_scores = cross_val_score(
    linear_regression, 
    X, 
    y, 
    cv=5, 
    scoring='r2'
)
train_scores = []
max_trees = [5, 10, 15, 20 ,25, 30]

for n in max_trees:
    model = GradientBoostingRegressor(
        n_estimators=n, 
        random_state=42
    )
    train_score =np.mean(cross_val_score(model, X, y, cv=5))
    train_scores.append(train_score)
    print(f"\nГрадиентный бустинг для  {n} деревьев: {train_score}")

print(f"\nЛинейная регрессия:{linear_regression_scores.mean()}")

for n in train_scores:
  if n>linear_regression_scores.mean():
    print("Градиентный бустинг лучше")
    break
  
#Графики зависимостей для наиболее информативных переменных

gradient_boosting = GradientBoostingRegressor()
linear_regression = LinearRegression()
gradient_boosting.fit(X, y)
linear_regression.fit(X, y)
top_7_variables = results.head(8)['Feature'].tolist()

fig, axs = plt.subplots(ncols=2, nrows=4, figsize=(20, 10))
axs = axs.flatten()

for idx, feature in enumerate(top_7_variables):
    axes = axs[idx]
    sns.lineplot(
        x=X[feature],
        y=linear_regression.predict(X),
        ax=axes, 
        color='orange', 
        label='Линейная регрессия'
    )
    sns.lineplot(
        x=X[feature], 
        y=gradient_boosting.predict(X),
        ax=axes, 
        color='blue', 
        label='Градиентный бустинг'
    )
    axes.set_title(f'Линейная регрессия vs Градиентный бустинг: {feature}', fontsize=10)
    axes.set_xlabel(feature)
    axes.set_ylabel('Price_euros')

plt.tight_layout()
plt.savefig('графики_зависимостей.png')
plt.show()
