import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import linalg
import matplotlib as mpl
from matplotlib import colors
from sklearn.datasets import load_iris
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay


# Загрузка данных Iris
data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target


# Анализ корреляций
correlation=df.corr()
print(f"\nКорреляционная матрица (все данные):\n{correlation}")
correlation_by_class = df.groupby('target').corr()
print(f"\nКорреляционная матрица по классам:\n{correlation_by_class}")
sns.pairplot(df, hue='target')
plt.savefig('графики_зависимостей.png')
plt.show()

# Линейный и квадратичный дискриминант

# Выбор признаков для анализа
name=['petal length (cm)',  'petal width (cm)']
X = df[name].values
y = data.target


def plot_data(lda, X, y, y_pred, fig_index, fir, sec):
    splot = plt.subplot(fir, sec, fig_index)
    if fig_index == 1:
        plt.title("Linear Discriminant Analysis")
    elif fig_index == 2:
        plt.title("Quadratic Discriminant Analysis")
    tp = y == y_pred  # True Positive
    tp0, tp1, tp2 = tp[y == 0], tp[y == 1], tp[y==2]
    X0, X1, X2 = X[y == 0], X[y == 1], X[y==2]
    X0_tp, X0_fp = X0[tp0], X0[~tp0]
    X1_tp, X1_fp = X1[tp1], X1[~tp1]
    X2_tp, X2_fp = X2[tp2], X2[~tp2]
    # класс 0: точки
    plt.scatter(X0_tp[:, 0], X0_tp[:, 1], marker=".", color="red")
    plt.scatter(X0_fp[:, 0], X0_fp[:, 1], marker="x", s=20, color="red")
    # класс 1: точки
    plt.scatter(X1_tp[:, 0], X1_tp[:, 1], marker=".", color="blue")
    plt.scatter(X1_fp[:, 0], X1_fp[:, 1], marker="x", s=20, color="blue")
    # класс 2: точки
    plt.scatter(X2_tp[:, 0], X2_tp[:, 1], marker=".", color="orange")
    plt.scatter(X2_fp[:, 0], X2_fp[:, 1], marker="x", s=20, color="orange")

    # классы 0, 1, 2: зоны
    nx, ny = 200, 100
    x_min, x_max = plt.xlim()
    y_min, y_max = plt.ylim()
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, nx), np.linspace(y_min, y_max, ny))
    Z = lda.predict_proba(np.c_[xx.ravel(), yy.ravel()])
    Z = Z[:, 1].reshape(xx.shape)
    plt.pcolormesh(
      xx, yy, Z, cmap='RdYlBu_r', 
      norm=colors.Normalize(0.0, 1.0), 
      zorder=0
    )
    plt.contour(xx, yy, Z, [0.5], linewidths=2.0, colors="white")
    return splot

def plot_ellipse(splot, mean, cov, color):
    v, w = linalg.eigh(cov)
    u = w[0] / linalg.norm(w[0])
    angle = np.arctan(u[1] / u[0])
    angle = 180 * angle / np.pi 
    ell = mpl.patches.Ellipse(
        mean,
        2 * v[0] ** 0.5,
        2 * v[1] ** 0.5,
        angle=180 + angle,
        facecolor=color,
        edgecolor="black",
        linewidth=2,
    )
    ell.set_clip_box(splot.bbox)
    ell.set_alpha(0.2)
    splot.add_artist(ell)
    splot.set_xticks(())
    splot.set_yticks(())
    
plt.figure(figsize=(10, 8), facecolor="white")

# Линейный дискриминант
lda = LinearDiscriminantAnalysis(solver="svd", store_covariance=True)
y_pred = lda.fit(X, y).predict(X)
splot = plot_data(lda, X, y, y_pred, fig_index=1, fir=2,sec=1)
plot_ellipse(splot, lda.means_[0], lda.covariance_, "red")
plot_ellipse(splot, lda.means_[1], lda.covariance_, "blue")
plot_ellipse(splot, lda.means_[2], lda.covariance_, "yellow")
plt.axis("tight")

# Квадратичный дискриминант
qda = QuadraticDiscriminantAnalysis(store_covariance=True)
y_pred = qda.fit(X, y).predict(X)
splot = plot_data(qda, X, y, y_pred, fig_index=2,fir=2,sec=1)
plot_ellipse(splot, qda.means_[0], qda.covariance_[0], "red")
plot_ellipse(splot, qda.means_[1], qda.covariance_[1], "blue")
plot_ellipse(splot, qda.means_[2], qda.covariance_[2], "yellow")
plt.axis("tight")

plt.tight_layout()
plt.subplots_adjust(top=0.92)
plt.savefig('дискриминанты.png')
plt.show()

# Логистическая регрессия
centers = [[-5, 0], [0, 1.5], [5, -1]]


for multi_class in ("multinomial", "ovr"):
    clf = LogisticRegression(
        solver="sag", 
        max_iter=100, 
        random_state=42, 
        multi_class=multi_class
    ).fit(X, y)

    print("training score : %.3f (%s)" % (clf.score(X, y), multi_class))

    _, ax = plt.subplots()
    DecisionBoundaryDisplay.from_estimator(
        clf, 
        X, 
        response_method="predict", 
        cmap=plt.cm.Paired, 
        ax=ax
    )
    plt.title("Decision surface of LogisticRegression (%s)" % multi_class)
    plt.axis("tight")

    colors = "bry"
    for i, color in zip(clf.classes_, colors):
        idx = np.where(y == i)
        plt.scatter(
            X[idx, 0], 
            X[idx, 1], 
            c=color, 
            cmap=plt.cm.Paired, 
            edgecolor="black", 
            s=20
        )

    # графики трех классификаторов «один против всех».
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    coef = clf.coef_
    intercept = clf.intercept_

    def plot_hyperplane(c, color):
        def line(x0):
            return (-(x0 * coef[c, 0]) - intercept[c]) / coef[c, 1]

        plt.plot([xmin, xmax], [line(xmin), line(xmax)], ls="--", color=color)

    for i, color in zip(clf.classes_, colors):
        plot_hyperplane(i, color)
    plt.savefig(f'{multi_class}.png')
plt.show()

# SVС
C = 1.0  # Параметр регуляризации SVС
models = (
    svm.SVC(kernel="linear", C=C),
    svm.SVC(kernel="poly", degree=2, gamma="auto", C=C),
)
models = (clf.fit(X, y) for clf in models)
titles = (
    "SVC with linear kernel",
    "SVC with polynomial (degree 2) kernel",
)


fig, sub = plt.subplots(2, 1)
plt.subplots_adjust(wspace=0.2, hspace=0.4)
X0, X1 = X[:, 0], X[:, 1]
for clf, title, ax in zip(models, titles, sub.flatten()):
    disp = DecisionBoundaryDisplay.from_estimator(
        clf,
        X,
        response_method="predict",
        cmap=plt.cm.coolwarm,
        alpha=0.8,
        ax=ax,
        xlabel=data.feature_names[2],
        ylabel=data.feature_names[3],
    )
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors="k")
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
plt.savefig('SVC.png')
plt.show()

#Квадратичная разделяющая функция 

petal_length = df['petal length (cm)']
petal_width = df['petal width (cm)']
target = df['target']

virginica_petal_length = petal_length[target == 2]
virginica_petal_width = petal_width[target == 2]
versicolor_petal_length = petal_length[target == 1]
versicolor_petal_width = petal_width[target == 1]

virginica_mean = np.array([virginica_petal_length.mean(), virginica_petal_width.mean()])
versicolor_mean = np.array([versicolor_petal_length.mean(), versicolor_petal_width.mean()])

virginica_covariance = np.cov(virginica_petal_length, virginica_petal_width)
versicolor_covariance = np.cov(versicolor_petal_length, versicolor_petal_width)

def quadratic_discriminant_function(x, mean, covariance):
    inv_covariance = np.linalg.inv(covariance)
    det_covariance = np.linalg.det(covariance)
    diff = x - mean
    return -0.5 * np.log(det_covariance) - 0.5 * np.dot(np.dot(diff, inv_covariance), diff)+np.log(0.5)

# Шаг 5: Визуализация разделяющей функции
x = np.linspace(1, 7, 100)
y = np.linspace(0, 2.5, 100)
X, Y = np.meshgrid(x, y)
Z_virginica = np.array([quadratic_discriminant_function(np.array([xi, yi]), virginica_mean, virginica_covariance) for xi, yi in zip(np.ravel(X), np.ravel(Y))])
Z_versicolor = np.array([quadratic_discriminant_function(np.array([xi, yi]), versicolor_mean, versicolor_covariance) for xi, yi in zip(np.ravel(X), np.ravel(Y))])
Z_virginica = Z_virginica.reshape(X.shape)
Z_versicolor = Z_versicolor.reshape(X.shape)

plt.contour(X, Y, Z_virginica, levels=[0], colors='blue', linewidths=2)
plt.contour(X, Y, Z_versicolor, levels=[0], colors='red', linewidths=2)
plt.scatter(virginica_petal_length, virginica_petal_width, c='blue', label='Virginica')
plt.scatter(versicolor_petal_length, versicolor_petal_width, c='red', label='Versicolor')

plt.xlabel('Petal Length (cm)')
plt.ylabel('Petal Width (cm)')
plt.title('Quadratic Discriminant Function')
plt.legend()
plt.savefig('Quadratic_Discriminant.png')
plt.show()