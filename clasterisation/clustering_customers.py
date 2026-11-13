"""
Кластеризация клиентов розничной сети (scikit-learn). Задания 1-12.
Запуск: python clustering_customers.py   (customers.csv должен лежать рядом).
Графики показываются по очереди: закрывайте окно, чтобы продолжить.
Выводы к каждому заданию приведены в комментариях.
"""

# # Кластеризация клиентов розничной сети (scikit-learn)
# Выполнение заданий 1–12 из учебного материала «Кластеризация данных в scikit-learn: теория и практические задания».
#
# Структура: A — данные и предобработка (задания 1–2) · B — K-Means (3–5) · C — иерархическая кластеризация, эталон, DBSCAN (6–8) · D — форма кластеров (9–10) · E — отчёт и исследование (11–12).
#
# Везде фиксирован random_state, поэтому результаты воспроизводимы (версия scikit-learn влияет на последние знаки).

import os, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.cluster.hierarchy import linkage, dendrogram

from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.datasets import make_blobs, make_moons
from sklearn.metrics import (silhouette_score, adjusted_rand_score,
                             normalized_mutual_info_score)

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 100
RS = 42
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)

# ## Блок A. Знакомство с данными и предобработка

# ### Задание 1. Загрузка и первичный анализ

DATA = next(p for p in ["customers.csv"] if os.path.exists(p))
df = pd.read_csv(DATA)

print("Размер таблицы:", df.shape)
print("\nТипы столбцов:")
print(df.dtypes)
print("\nПропуски по столбцам:")
print(df.isna().sum())
print(df.describe().T.round(2))

num_cols = [c for c in df.columns if c != "customer_id"]
fig, axes = plt.subplots(2, 4, figsize=(16, 7))
for ax, col in zip(axes.ravel(), num_cols):
    sns.histplot(df[col], bins=30, ax=ax, color="#4C72B0")
    ax.set_title(col)
axes.ravel()[-1].axis("off")
plt.tight_layout(); plt.show()

sns.pairplot(df[["age", "monthly_income_k", "avg_check", "online_share"]].dropna(),
             corner=True, plot_kws={"s": 12, "alpha": 0.6})
plt.show()

# Масштаб признаков и кандидаты в выбросы
scale = df[num_cols].agg(["min", "max", "std"]).T
scale["диапазон"] = scale["max"] - scale["min"]
print(scale.sort_values("std", ascending=False).round(2))

# Выбросы: |z| > 4 по любому признаку (пропуски пока игнорируем)
z = (df[num_cols] - df[num_cols].mean()) / df[num_cols].std()
suspicious = (z.abs() > 4).any(axis=1)
print("\nКлиентов с |z| > 4 хотя бы по одному признаку:", suspicious.sum())
print(df.loc[suspicious, ["customer_id", "age", "monthly_income_k", "avg_check", "tenure_months"]])

# Ответы на вопросы задания 1
#
# *Масштаб.* Признаки различаются на порядки: avg_check лежит в диапазоне 260–14 990 руб. (std ≈ 1 950), monthly_income_k — 18,7–304 тыс. (std ≈ 48), tenure_months — 1–137, spending_score — 3–100, age — 18–76, visits_per_month — 0–29, а online_share вообще от 0 до 1 (std ≈ 0,25). Без масштабирования K-Means «видел» бы почти только avg_check и доход.
#
# *Выбросы.* Да, они очевидны: ровно 10 клиентов имеют экстремальные значения (|z| > 4): средний чек примерно 9 000–15 000 руб. и/или доход 230–300 тыс. при медианах около 1 850 руб. и 77 тыс. Это хорошо видно на гистограммах avg_check и monthly_income_k (длинный правый хвост) и на парных диаграммах (одиночные точки далеко справа/вверху).
#
# *Пропуски.* 15 пропусков в monthly_income_k (2,5 %); заполним медианой — она устойчива к выбросам.
#
# *Корреляции.* Сильно связаны возраст, online_share и срок в программе (старшие клиенты — дольше с нами и реже покупают онлайн), а также доход и средний чек — признаки частично дублируют друг друга, что важно для PCA (задание 5).

# ### Задание 2. Pipeline: импутация + масштабирование

features = df.drop(columns="customer_id")          # идентификатор в кластеризации не используем
prep = make_pipeline(SimpleImputer(strategy="median"), StandardScaler())
X = prep.fit_transform(features)

print("X.shape =", X.shape, "| пропусков в X:", int(np.isnan(X).sum()))
check = pd.DataFrame({"mean": X.mean(axis=0), "std": X.std(axis=0)}, index=features.columns)
print(check.round(4))
assert np.allclose(X.mean(axis=0), 0, atol=1e-8) and np.allclose(X.std(axis=0), 1)

# Среднее всех признаков равно 0, стандартное отклонение — 1: предобработка выполнена корректно. customer_id исключён, меток в данных нет, поэтому утечки информации нет. Обучать prep на всей таблице допустимо: мы не оцениваем предсказательную модель на отложенной выборке, а только описываем имеющихся клиентов.

# ## Блок B. K-Means

# ### Задание 3. Выбор числа кластеров: локоть и силуэт

ks = list(range(2, 11))
inertias, sils = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RS).fit(X)
    inertias.append(km.inertia_)
    sils.append(silhouette_score(X, km.labels_))

res = pd.DataFrame({"k": ks, "inertia": inertias, "silhouette": sils}).set_index("k")
res["падение inertia"] = res["inertia"].diff().round(1)
print(res.round(3))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(ks, inertias, marker="o"); ax[0].set_title("Inertia (метод локтя)")
ax[0].set_xlabel("k"); ax[0].set_ylabel("inertia")
ax[1].plot(ks, sils, marker="o", color="darkorange"); ax[1].set_title("Коэффициент силуэта")
ax[1].set_xlabel("k"); ax[1].set_ylabel("silhouette")
for a in ax: a.axvline(6, ls="--", c="gray", lw=1)
plt.tight_layout(); plt.show()

# Вывод: выбираю k = 6.
#
# * Локоть. Inertia быстро падает до k = 6 (2812 → 1992 → 1480 → 1204 → 950), а затем падение резко замедляется (950 → 898 → 861). Излом — в районе k = 6.
# * Силуэт. Растёт до k = 4 (0,383), слегка снижается на k = 5 (0,361) и достигает глобального максимума на k = 6 (0,393); дальше устойчиво падает. Значения k = 4 и k = 6 почти равны, т. е. однозначного победителя нет — как и предупреждает теория.
# * Содержательный критерий (решающий). Профили кластеров (задание 4) показывают, что при k = 6 один из кластеров состоит ровно из 10 клиентов-выбросов, а остальные пять — понятные покупательские сегменты. При k = 4 «слипаются» разные группы, при k = 5 выбросы всё равно занимают отдельный кластер, и «настоящих» сегментов остаётся только четыре. Поэтому k = 6 = 5 сегментов + кластер аномальных клиентов.

# ### Задание 4. Итоговая модель K-Means и профили кластеров

K = 6
km = KMeans(n_clusters=K, n_init=10, random_state=RS).fit(X)
df["cluster"] = km.labels_

profile = df.drop(columns="customer_id").groupby("cluster").mean()
profile["n"] = df.groupby("cluster").size()
print(profile.round(2))

def name_segments(prof, outlier_cluster=None):
    """Присваивает названия по профилю центроида (устойчиво к перенумерации кластеров)."""
    names, rest = {}, list(prof.index)
    if outlier_cluster is not None:
        names[outlier_cluster] = "Аномальные клиенты (выбросы)"
        rest.remove(outlier_cluster)
    p = prof.loc[rest]
    young = p["age"].idxmin();   names[young] = "Молодые экономные онлайн-покупатели"; p = p.drop(young)
    old = p["age"].idxmax();     names[old] = "Пожилые офлайн-клиенты с низким доходом"; p = p.drop(old)
    rare = p["visits_per_month"].idxmin(); names[rare] = "Обеспеченные, но редкие покупатели"; p = p.drop(rare)
    act = p["avg_check"].idxmax(); names[act] = "Обеспеченные активные с высоким чеком"; p = p.drop(act)
    names[p.index[0]] = "Семейный средний сегмент"
    return names

outlier_cluster = profile["n"].idxmin()
names = name_segments(profile.drop(columns="n"), outlier_cluster)
df["segment"] = df["cluster"].map(names)

summary = profile.rename(index=names).sort_values("n", ascending=False)
print(summary.round(2))

# Названия и описание кластеров (приблизительные средние значения из таблицы выше):
#
# 1. Молодые экономные онлайн-покупатели (≈150 клиентов). Средний возраст ≈ 23 года, доход ≈ 40 тыс., онлайн-доля ≈ 0,80, чек всего ≈ 870 руб., но заходят чаще всех (≈ 10 визитов в месяц). Недавние участники программы (≈ 13 мес.). Много маленьких покупок.
# 2. Семейный средний сегмент (≈140). Возраст ≈ 40, доход ≈ 75 тыс., чек ≈ 2 500 руб., ≈ 5 визитов, онлайн ≈ 0,45. Ничем не выделяется — «ядро» базы.
# 3. Обеспеченные активные с высоким чеком (≈120). Возраст ≈ 35, доход ≈ 130 тыс., индекс трат ≈ 80, средний чек ≈ 4 400 руб. Самый ценный сегмент по выручке.
# 4. Обеспеченные, но редкие покупатели (≈110). Возраст ≈ 46, доход ≈ 123 тыс., но индекс трат ≈ 26 и всего ≈ 2 визита в месяц; высокий потенциал, который не реализован.
# 5. Пожилые офлайн-клиенты с низким доходом (≈70). Возраст ≈ 63, доход ≈ 30 тыс., онлайн-доля ≈ 0,10, срок в программе ≈ 80 мес. — самые «старые» и лояльные, но малобюджетные.
# 6. Аномальные клиенты (выбросы) (10). Доход ≈ 270 тыс. и средний чек ≈ 12 000 руб. — не сегмент, а набор экстремальных значений; их нужно разбирать вручную (VIP, ошибки данных, B2B-покупатели?).

# ### Задание 5. Визуализация: PCA в 2D

pca = PCA(n_components=2, random_state=RS)
X2 = pca.fit_transform(X)
centers2 = pca.transform(km.cluster_centers_)       # центры — тем же PCA

fig, ax = plt.subplots(figsize=(9, 6.5))
palette = sns.color_palette("tab10", K)
for i, (cl, name) in enumerate(sorted(names.items(), key=lambda t: t[1])):
    m = df["cluster"].values == cl
    ax.scatter(X2[m, 0], X2[m, 1], s=22, alpha=0.7, color=palette[i], label=f"{name} (n={m.sum()})")
    ax.scatter(*centers2[cl], marker="X", s=260, color=palette[i], edgecolor="black", linewidth=1.5)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} дисперсии)")
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} дисперсии)")
ax.set_title("K-Means (k=6) в проекции на две главные компоненты; X — центры кластеров")
ax.legend(fontsize=8, loc="best"); plt.tight_layout(); plt.show()

print("Объяснённая дисперсия PC1 + PC2: %.1f%%" % (100 * pca.explained_variance_ratio_.sum()))
print("Для справки, три компоненты: %.1f%%" % (100 * PCA(3, random_state=RS).fit(X).explained_variance_ratio_.sum()))

# Две первые главные компоненты объясняют ≈ 72 % дисперсии (PC1 ≈ 45 %, PC2 ≈ 27 %), три — ≈ 82 %. Этого достаточно, чтобы картинка была информативной, но около 28 % информации на ней не видно, поэтому часть перекрытий между кластерами — артефакт проекции. Выбросы лежат далеко от основного облака и хорошо отделены.

# ## Блок C. Другие алгоритмы

# ### Задание 6. Иерархическая кластеризация

# Дендрограмма на случайной подвыборке из 120 объектов
rng = np.random.RandomState(RS)
idx = rng.choice(len(X), 120, replace=False)
Z = linkage(X[idx], method="ward")
cut = (Z[-K, 2] + Z[-(K - 1), 2]) / 2        # высота, на которой остаётся K кластеров

fig, ax = plt.subplots(figsize=(13, 5))
dendrogram(Z, no_labels=True, color_threshold=cut, ax=ax)
ax.axhline(cut, ls="--", c="black", lw=1, label=f"разрез на {K} кластеров")
ax.set_title("Дендрограмма (ward), подвыборка 120 клиентов"); ax.set_ylabel("расстояние")
ax.legend(); plt.tight_layout(); plt.show()

# Сравнение ward и K-Means при том же числе кластеров (K = 6)
ward = AgglomerativeClustering(n_clusters=K, linkage="ward").fit_predict(X)
print("ARI(ward, K-Means) = %.3f" % adjusted_rand_score(km.labels_, ward))
print("\nТаблица сопряжённости (строки — K-Means, столбцы — ward):")
print(pd.crosstab(pd.Series(km.labels_, name="K-Means").map(names), pd.Series(ward, name="ward")))

# Другие способы объединения
rows = []
for link in ["ward", "complete", "average", "single"]:
    lab = AgglomerativeClustering(n_clusters=K, linkage=link).fit_predict(X)
    rows.append({"linkage": link,
                 "размеры кластеров": sorted(np.bincount(lab), reverse=True),
                 "silhouette": round(silhouette_score(X, lab), 3),
                 "ARI с K-Means": round(adjusted_rand_score(km.labels_, lab), 3)})
print(pd.DataFrame(rows).set_index("linkage"))

# Сравнение. Ward почти полностью повторяет K-Means: ARI = 0,958, выбросы выделены в тот же кластер из 10 объектов, а расхождения — всего около десятка клиентов на границах пересекающихся сегментов («семейные» ↔ «обеспеченные, но редкие», см. таблицу сопряжённости). Оба метода минимизируют внутрикластерную дисперсию, поэтому результат и получается близким.
#
# Другие linkage (k = 6).
# * complete даёт неравномерные кластеры — [367, 153, 70, 6, 3, 1]: он чувствителен к самым далёким точкам, то есть к выбросам, и дробит их на крошечные группы (ARI с K-Means ≈ 0,46).
# * average делит основную массу всего на два крупных кластера (407 и 183), а остальные четыре — это 1–3 выброса (ARI ≈ 0,33).
# * single — «цепочечный» эффект в чистом виде: один гигантский кластер из 590 объектов и пять крошечных из выбросов (ARI ≈ 0,02). Силуэт у него самый высокий (≈ 0,55), но это обманчиво: метрика «хвалит» разбиение, где все нормальные клиенты слиты в одну массу, а для сегментации оно бесполезно.
#
# Вывод: для сегментации клиентов подходит ward; single/average пригодны скорее как детектор выбросов.

# ### Задание 7. Проверка с эталоном (ARI и NMI)

# Файл customers_answers_teacher.csv не входил в загруженные материалы.
# Ячейка сама подхватит его, если положить рядом с ноутбуком (колонка true_segment).
candidates = ["customers_answers_teacher.csv", "/mnt/user-data/uploads/customers_answers_teacher.csv"]
path = next((p for p in candidates if os.path.exists(p)), None)

if path is None:
    print("Файл с эталоном не найден — положите customers_answers_teacher.csv рядом с ноутбуком и перезапустите ячейку.")
else:
    truth = pd.read_csv(path)
    y_true = truth["true_segment"].values
    rows = {}
    models = {
        "K-Means (k=6)": km.labels_,
        "K-Means (k=5)": KMeans(5, n_init=10, random_state=RS).fit_predict(X),
        "Ward (k=6)": ward,
        "Ward (k=5)": AgglomerativeClustering(5, linkage="ward").fit_predict(X),
    }
    for name, lab in models.items():
        rows[name] = {"ARI": adjusted_rand_score(y_true, lab),
                      "NMI": normalized_mutual_info_score(y_true, lab)}
    print(pd.DataFrame(rows).T.round(3))

# Что можно сказать без файла. Файл customers_answers_teacher.csv не был приложен, поэтому числа ARI/NMI я не приводил — подставлять их «по памяти» из методички было бы неправильно. Ячейка выше посчитает их автоматически, как только файл появится рядом с ноутбуком.
#
# Ожидаемый ответ на теоретический вопрос «почему ARI ≠ 1»: (1) сегменты частично пересекаются по признакам (это видно на PCA-диаграмме), поэтому часть клиентов на границах неизбежно попадает не в «свой» кластер; (2) выбросы искажают центроиды и забирают себе отдельный кластер, из-за чего границы остальных сдвигаются; (3) в эталоне выбросы помечены как -1, и при k = 5 алгоритм не знает об этом отдельном классе. Ещё один полезный ориентир, который можно посчитать уже сейчас: согласованность K-Means и Ward между собой — ARI(ward, K-Means) из задания 6.

# ### Задание 8. DBSCAN на данных клиентов

# График k-расстояний (k = min_samples = 8)
MS = 8
dist, _ = NearestNeighbors(n_neighbors=MS).fit(X).kneighbors(X)
kdist = np.sort(dist[:, -1])

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(kdist); ax[0].set_title(f"Расстояние до {MS}-го соседа (отсортировано)")
ax[0].set_xlabel("объекты"); ax[0].set_ylabel("расстояние")
for e in (0.8, 1.5): ax[0].axhline(e, ls="--", c="gray", lw=1)
ax[1].plot(kdist); ax[1].set_ylim(0, 3); ax[1].set_xlim(450, 600)
ax[1].set_title("Увеличенный хвост"); plt.tight_layout(); plt.show()
print("Перцентили 50/80/90/95/99:", np.percentile(kdist, [50, 80, 90, 95, 99]).round(2))

sus = (np.abs(X) > 4).any(axis=1)          # 10 «подозрительных» клиентов из задания 1
rows = []
for eps in np.round(np.arange(0.8, 1.55, 0.1), 1):
    lab = DBSCAN(eps=eps, min_samples=MS).fit_predict(X)
    noise = lab == -1
    rows.append({"eps": eps,
                 "кластеров": len(set(lab) - {-1}),
                 "шум, шт.": int(noise.sum()),
                 "шум, %": round(100 * noise.mean(), 1),
                 "подозрительных среди шума (из 10)": int((noise & sus).sum())})
db_table = pd.DataFrame(rows).set_index("eps")
print(db_table)

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, eps in zip(axes, (0.8, 1.2, 1.5)):
    lab = DBSCAN(eps=eps, min_samples=MS).fit_predict(X)
    ax.scatter(X2[lab != -1, 0], X2[lab != -1, 1], c=lab[lab != -1], cmap="tab10", s=14)
    ax.scatter(X2[lab == -1, 0], X2[lab == -1, 1], c="black", marker="x", s=25, label="шум (-1)")
    ax.set_title(f"eps = {eps}: кластеров {len(set(lab) - {-1})}, шум {(lab == -1).sum()}")
    ax.legend()
plt.tight_layout(); plt.show()

# Выводы по DBSCAN.
#
# * По графику k-расстояний основная масса точек лежит ниже ≈ 1,4 (95-й перцентиль ≈ 1,43), а резкий излом идёт после ≈ 1,5; поэтому диапазон 0,8–1,5 разумен.
# * Результат очень чувствителен к eps: при 0,8 — 7 кластеров и ≈ 41 % точек в шуме; при 1,0 — 3 кластера и ≈ 80 шумовых точек; при 1,2 — 2 кластера и ≈ 20 шумовых точек; при 1,5 остаётся один кластер и шум = 10 точек.
# * Метку -1 получают, во-первых, все 10 «подозрительных» клиентов из задания 1 (они в шуме при любом eps из таблицы), и, во-вторых, при малых eps — клиенты на «разреженных» краях сегментов. При eps = 1,5 шум в точности совпадает с выбросами — DBSCAN работает как отличный детектор аномалий.
# * Но как алгоритм сегментации он здесь слабее K-Means: сегменты пересекаются и имеют разную плотность, поэтому, когда eps достаточно велик, чтобы не выбрасывать пол-выборки, кластеры сливаются в одну массу. Это прямая иллюстрация слабости DBSCAN, описанной в теории.

# ## Блок D. Форма кластеров

# ### Задание 9. make_moons: K-Means vs Agglomerative vs DBSCAN

M, y_m = make_moons(n_samples=400, noise=0.07, random_state=0)

# подбор eps для DBSCAN
for eps in (0.1, 0.15, 0.2, 0.3):
    l = DBSCAN(eps=eps, min_samples=5).fit_predict(M)
    print(f"eps={eps}: кластеров {len(set(l) - {-1})}, шум {(l == -1).sum()}, ARI={adjusted_rand_score(y_m, l):.3f}")

results = {
    "K-Means (k=2)": KMeans(2, n_init=10, random_state=0).fit_predict(M),
    "Agglomerative ward (k=2)": AgglomerativeClustering(2, linkage="ward").fit_predict(M),
    "DBSCAN (eps=0.2, min_samples=5)": DBSCAN(eps=0.2, min_samples=5).fit_predict(M),
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.3))
for ax, (name, lab) in zip(axes, results.items()):
    ax.scatter(M[:, 0], M[:, 1], c=lab, cmap="coolwarm", s=18)
    ax.set_title(f"{name}\nARI = {adjusted_rand_score(y_m, lab):.2f}")
plt.tight_layout(); plt.show()

# K-Means и ward разрезают «луны» по прямой (ARI ≈ 0,27 и 0,24): K-Means ищет выпуклые «круглые» кластеры вокруг центров, поэтому для полумесяцев сумма квадратов расстояний минимальна именно при таком неправильном разбиении. DBSCAN с eps = 0,2 восстанавливает обе луны идеально (ARI = 1), потому что определяет кластер как связную область высокой плотности и не требует выпуклой формы.
#
# Плотностные методы предпочтительнее, когда кластеры имеют произвольную (невыпуклую, вытянутую, вложенную) форму, когда есть шум/выбросы и когда число кластеров заранее неизвестно. Цена — чувствительность к eps (при 0,1 луны распадаются на 5 кусков, при 0,3 сливаются в один).

# ### Задание 10. make_blobs: K-Means vs GaussianMixture

B, y_b, centers_b = make_blobs(n_samples=600, centers=3, cluster_std=[0.5, 1.5, 0.8],
                               random_state=1, return_centers=True)
km_b = KMeans(3, n_init=10, random_state=1).fit(B)
gm_b = GaussianMixture(3, random_state=1).fit(B)
lab_km, lab_gm = km_b.labels_, gm_b.predict(B)

print("ARI K-Means:        %.3f" % adjusted_rand_score(y_b, lab_km))
print("ARI GaussianMixture: %.3f" % adjusted_rand_score(y_b, lab_gm))

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (title, lab) in zip(axes, [("Истинные кластеры", y_b), ("K-Means", lab_km), ("GaussianMixture", lab_gm)]):
    ax.scatter(B[:, 0], B[:, 1], c=lab, cmap="viridis", s=14); ax.set_title(title)
plt.tight_layout(); plt.show()

# Мягкая кластеризация: объекты с самой неуверенной принадлежностью
proba = gm_b.predict_proba(B)
border = np.argsort(proba.max(axis=1))[:6]
print(pd.DataFrame(proba[border], columns=[f"P(кластер {i})" for i in range(3)]).round(3))
print("\nТочек, у которых максимальная вероятность < 0.9:", int((proba.max(axis=1) < 0.9).sum()))

# Разные размеры кластеров (300 / 100 / 200 объектов) при тех же центрах и разбросе
B2, y_b2 = make_blobs(n_samples=[300, 100, 200], centers=centers_b,
                      cluster_std=[0.5, 1.5, 0.8], random_state=1)
print("\nНеравные размеры кластеров:")
print("  ARI K-Means:         %.3f" % adjusted_rand_score(y_b2, KMeans(3, n_init=10, random_state=1).fit_predict(B2)))
print("  ARI GaussianMixture: %.3f" % adjusted_rand_score(y_b2, GaussianMixture(3, random_state=1).fit_predict(B2)))

# При разном разбросе кластеров K-Means ошибается на границах: он проводит границу посередине между центрами (равное расстояние), поэтому «широкий» кластер с std = 1,5 теряет свои периферийные точки в пользу соседей. GaussianMixture оценивает для каждого кластера собственную дисперсию и вес, поэтому границы смещаются в сторону «узкого» кластера, и разбиение почти совпадает с истинным (ARI ≈ 0,99 против ≈ 0,90 у K-Means).
#
# predict_proba показывает «серую зону»: у граничных объектов вероятности принадлежности двум кластерам сопоставимы (например, 0,4 / 0,6) — жёсткие метки K-Means эту неопределённость скрывают. Это полезно, когда клиента нельзя уверенно отнести к одному сегменту. При неравных размерах кластеров (300/100/200) разрыв сокращается (ARI ≈ 0,96 у K-Means и ≈ 0,99 у GaussianMixture), но GaussianMixture всё равно точнее.

# ## Блок E. Исследовательское задание

# ### Задание 11. Итоговое разбиение и мини-отчёт

# Финальный вариант: выбросы выделяем явно (|z| > 4 ↔ шум DBSCAN при eps=1.5), остальных клиентов делим K-Means на 5 сегментов
db_noise = DBSCAN(eps=1.5, min_samples=MS).fit_predict(X) == -1
print("Выбросы по |z|>4 и шум DBSCAN совпадают полностью:", bool((db_noise == sus).all()))

Xc = X[~sus]
sil_clean = {k: round(silhouette_score(Xc, KMeans(k, n_init=10, random_state=RS).fit_predict(Xc)), 3) for k in range(2, 9)}
print("Силуэт на очищенных данных:", sil_clean)

km_c = KMeans(5, n_init=10, random_state=RS).fit(Xc)
final = df.loc[~sus].copy()
final["cluster"] = km_c.labels_
prof_c = final.drop(columns=["customer_id", "segment"]).groupby("cluster").mean()
prof_c["n"] = final.groupby("cluster").size()
names_c = name_segments(prof_c.drop(columns="n"))
final["segment"] = final["cluster"].map(names_c)

print("\nARI между «k=6 на всех данных» и «k=5 без выбросов» (на 590 общих клиентах): %.3f"
      % adjusted_rand_score(df.loc[~sus, "cluster"], final["cluster"]))
print()
rep = prof_c.rename(index=names_c).sort_values("n", ascending=False)
rep["доля, %"] = (100 * rep["n"] / len(df)).round(1)
print(rep.round(2))

# ## Мини-отчёт: сегментация клиентов розничной сети
#
# Выбор числа кластеров. Выбрано 5 сегментов + отдельная группа из 10 аномальных клиентов (эквивалентно K-Means с k = 6 на всех данных). Основания: (1) на полной выборке силуэт максимален при k = 6 (0,393), а излом метода локтя тоже приходится на k = 6; (2) после удаления выбросов силуэт максимален при k = 5 (0,394) — то есть оба подхода дают один и тот же набор сегментов; (3) разбиение содержательно интерпретируется и устойчиво (ARI между вариантами ≈ 1, ward и K-Means на очищенных данных согласуются).
#
# Сегменты и маркетинговые действия
#
# | Сегмент | Доля | Профиль | Действие |
# |---|---|---|---|
# | Молодые экономные онлайн-покупатели | ≈ 25 % | ≈ 23 года, доход ≈ 40 тыс., чек ≈ 870 руб., ≈ 10 визитов, онлайн 80 % | Мобильное приложение, push-акции на частые мелкие покупки, программа «купи N — получи подарок», подписка с бесплатной доставкой |
# | Семейный средний сегмент | ≈ 24 % | ≈ 40 лет, доход ≈ 75 тыс., чек ≈ 2 500 руб. | Семейные наборы и «корзины недели», кросс-продажи, накопительные баллы |
# | Обеспеченные активные с высоким чеком | ≈ 20 % | ≈ 35 лет, доход ≈ 130 тыс., индекс трат ≈ 80, чек ≈ 4 400 руб. | Удержание: премиальный статус, ранний доступ к новинкам, персональный менеджер |
# | Обеспеченные, но редкие покупатели | ≈ 19 % | ≈ 46 лет, доход ≈ 123 тыс., ≈ 2 визита, индекс трат ≈ 26 | Реактивация: персональные предложения премиум-товаров, напоминания и купоны «вернись», опрос о причинах редких покупок |
# | Пожилые офлайн-клиенты с низким доходом | ≈ 12 % | ≈ 63 года, доход ≈ 30 тыс., онлайн ≈ 10 %, стаж ≈ 80 мес. | Офлайн-акции и скидки по карте лояльности, дни пенсионера, помощь с освоением онлайн-заказа |
# | Аномальные клиенты | ≈ 1,7 % (10 чел.) | чек 9–15 тыс., доход до 300 тыс. | Ручной разбор: проверить на ошибки данных/B2B, при подтверждении — VIP-обслуживание |
#
# Ограничения анализа.
# * Данные синтетические, а метрики качества (силуэт ≈ 0,39) говорят о перекрывающихся сегментах: границы условны, часть клиентов лежит «между» группами.
# * Выбор k неоднозначен: k = 4 и k = 6 по силуэту почти не различаются; итог опирается на интерпретируемость.
# * 15 пропусков в доходе заполнены медианой, что слегка сужает разброс.
# * Выбросы определены по порогу |z| > 4 — порог произвольный; единственная причина их отделения — их влияние на центроиды.
# * Кластеризация — описательная; эффект маркетинговых действий нужно проверять A/B-тестами. Сегменты нужно пересчитывать при появлении новых данных.

# ### Задание 12 (по желанию). Роль инициализации и масштабирования

ref = KMeans(5, n_init=10, random_state=RS).fit(X)          # опорное разбиение (минимум inertia)
rows = []
for n_init in (1, 10):
    for seed in range(20):
        m = KMeans(5, n_init=n_init, random_state=seed).fit(X)
        rows.append({"n_init": n_init, "seed": seed, "inertia": m.inertia_,
                     "ARI с опорным": adjusted_rand_score(ref.labels_, m.labels_)})
rep12 = pd.DataFrame(rows)
print(rep12.groupby("n_init").agg(inertia_min=("inertia", "min"), inertia_max=("inertia", "max"),
                                  ARI_min=("ARI с опорным", "min"), ARI_mean=("ARI с опорным", "mean")).round(3))

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
sns.stripplot(data=rep12, x="n_init", y="inertia", ax=ax[0], jitter=0.2)
ax[0].set_title("Inertia по 20 запускам (k=5)")
sns.stripplot(data=rep12, x="n_init", y="ARI с опорным", ax=ax[1], jitter=0.2)
ax[1].set_title("ARI относительно лучшего разбиения")
plt.tight_layout(); plt.show()

# Что будет без масштабирования
X_raw = SimpleImputer(strategy="median").fit_transform(features)
km_raw = KMeans(5, n_init=10, random_state=RS).fit(X_raw)
print("ARI(без масштабирования, со масштабированием) = %.3f" % adjusted_rand_score(ref.labels_, km_raw.labels_))
print("Размеры кластеров без масштабирования:", [int(v) for v in sorted(np.bincount(km_raw.labels_), reverse=True)])
print("Размеры кластеров с масштабированием: ", [int(v) for v in sorted(np.bincount(ref.labels_), reverse=True)])

raw_prof = pd.DataFrame(X_raw, columns=features.columns).groupby(km_raw.labels_).mean()
print("\nСредние по кластерам без масштабирования:")
print(raw_prof.round(1))

# Выводы.
#
# * Инициализация. При n_init=1 результат зависит от seed: inertia колеблется от ≈ 1203 до ≈ 1378 (в худших запусках алгоритм застревает в плохом локальном минимуме), а ARI с лучшим разбиением в худшем случае падает до ≈ 0,56 (в среднем ≈ 0,80). При n_init=10 почти все запуски приходят к одному оптимуму: inertia ≈ 1203–1204, ARI ≈ 0,98–1. Значит, n_init > 1 (и k-means++) нужен всегда, а результат одного запуска нельзя считать окончательным.
# * Масштабирование. Без него разбиение становится совсем другим (ARI с корректным ≈ 0,37): в евклидовых расстояниях доминируют avg_check (десятки тысяч) и доход, а online_share (0–1), возраст и визиты практически не учитываются. Кластеры расходятся лишь по чеку и доходу, а профили теряют смысл (онлайн-доля и возраст перестают разделять сегменты).
