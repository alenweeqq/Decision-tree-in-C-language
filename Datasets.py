from sklearn.datasets import make_classification
import pandas as pd

# генерируем датасет
X, y = make_classification(
    n_samples=500,        # строк
    n_features=4,         # признаков (как у тебя)
    n_informative=3,      # полезные признаки
    n_redundant=1,        # шум/зависимость
    n_classes=4,          # классы
    n_clusters_per_class=1,
    class_sep=1.0,        # НАСКОЛЬКО хорошо разделимы классы (1.0 = нормально)
    random_state=42
)

# собираем в таблицу
df = pd.DataFrame(X, columns=["f1", "f2", "f3", "f4"])
df["label"] = y

# сохраняем
df.to_csv("dataset.csv", index=False)

print(df.head())