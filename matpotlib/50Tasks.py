import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages
import os
import time

df = pd.read_csv("Internet Speed 2022.csv")
mobile_raw = df['mobile'].copy()
df['mobile'] = df['mobile'].fillna(df['mobile'].median())
rng = np.random.default_rng(42)

# 1
x = np.linspace(0, 4 * np.pi, 200)
plt.plot(x, np.sin(x), color='blue', linestyle='-', label='sin(x)')
plt.plot(x, np.cos(x), color='red', linestyle='--', label='cos(x)')
plt.legend()
plt.grid(True)
plt.show()

# 2
x2 = np.linspace(-2, 2, 100)
plt.plot(x2, x2, marker='o', markevery=20, label='x')
plt.plot(x2, x2**2, marker='s', markevery=20, label='x^2')
plt.plot(x2, x2**3, marker='^', markevery=20, label='x^3')
plt.xlim(-2, 2)
plt.ylim(-10, 10)
plt.legend()
plt.show()

# 3
t = np.linspace(0, 5, 200)
plt.plot(t, np.exp(-t) * np.cos(5*t), label='e^(-t)cos(5t)')
plt.plot(t, np.exp(-t), 'k--', label='e^(-t)')
plt.plot(t, -np.exp(-t), 'k--')
plt.legend()
plt.show()

# 4
for _ in range(3):
    plt.plot(np.cumsum(rng.normal(size=500)), alpha=0.7)
plt.show()

# 5
df_sorted = df.sort_values(by='broadband', ascending=False).reset_index()
plt.plot(df_sorted.index, df_sorted['broadband'])
plt.yscale('log')
plt.xlabel('Место в рейтинге')
plt.ylabel('Скорость (log)')
plt.show()

# 6
height = rng.normal(170, 10, 100)
weight = height * 0.4 + rng.normal(0, 5, 100)
corr = np.corrcoef(height, weight)[0, 1]
plt.scatter(height, weight)
plt.title(f'Корреляция: {corr:.2f}')
plt.show()

# 7
z = rng.uniform(0, 100, 100)
sc = plt.scatter(height, weight, c=z, cmap='viridis')
plt.colorbar(sc)
plt.show()

# 8
x_city = rng.uniform(0, 10, 30)
y_city = rng.uniform(0, 10, 30)
pop = rng.uniform(10, 1000, 30)
plt.scatter(x_city, y_city, s=pop, alpha=0.5)
plt.show()

# 9
plt.scatter(df['broadband'], df['mobile'])
z_trend = np.polyfit(df['broadband'], df['mobile'], 1)
p = np.poly1d(z_trend)
plt.plot(df['broadband'], p(df['broadband']), "r--")
plt.show()

# 10
top5_mobile = df.nlargest(5, 'mobile')
plt.scatter(df['broadband'], df['mobile'])
for _, row in top5_mobile.iterrows():
    plt.annotate(row['country'], (row['broadband'], row['mobile']))
plt.show()

# 11
norm_data = rng.normal(size=1000)
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, b in zip(axes, [10, 20, 50]):
    ax.hist(norm_data, bins=b)
    ax.set_title(f'bins={b}')
plt.show()

# 12
plt.hist(rng.normal(0, 1, 500), alpha=0.5, label='Mean=0')
plt.hist(rng.normal(3, 1, 500), alpha=0.5, label='Mean=3')
plt.legend()
plt.show()

# 13
plt.hist(df['broadband'], bins=20)
plt.axvline(df['broadband'].mean(), color='red', label='Среднее')
plt.axvline(df['broadband'].median(), color='green', label='Медиана')
plt.legend()
plt.show()

# 14
fig, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
ax1.hist(df['broadband'], bins=15)
ax1.set_title('Broadband')
ax2.hist(df['mobile'], bins=15)
ax2.set_title('Mobile')
plt.show()

# 15
langs = ['Python', 'Java', 'C++', 'JS', 'C#']
shares = [30, 20, 15, 10, 5]
bars = plt.bar(langs, shares)
plt.bar_label(bars)
plt.show()

# 16
plt.barh(langs, shares)
plt.show()

# 17
cat = np.arange(4)
y21 = [10, 20, 15, 25]
y22 = [12, 22, 18, 28]
plt.bar(cat - 0.2, y21, width=0.4, label='2021')
plt.bar(cat + 0.2, y22, width=0.4, label='2022')
plt.xticks(cat, ['A', 'B', 'C', 'D'])
plt.legend()
plt.show()

# 18
top10_bb = df.nlargest(10, 'broadband').sort_values('broadband', ascending=True)
plt.barh(top10_bb['country'], top10_bb['broadband'])
plt.show()

# 19
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
worst_bb = df.nsmallest(10, 'broadband')
best_mob = df.nlargest(10, 'mobile')
ax1.barh(worst_bb['country'], worst_bb['broadband'])
ax1.set_title('Худший broadband')
ax2.barh(best_mob['country'], best_mob['mobile'])
ax2.set_title('Лучший mobile')
plt.tight_layout()
plt.show()

# 20
budget = [40, 20, 20, 10, 10]
labels = ['Жилье', 'Еда', 'Транспорт', 'Развлечения', 'Прочее']
plt.pie(budget, labels=labels, autopct='%1.1f%%', startangle=90, explode=[0.1, 0, 0, 0, 0])
plt.show()

# 21
df['bb_group'] = pd.cut(df['broadband'], bins=[0, 25, 100, float('inf')], labels=['<25', '25-100', '>100'])
df['bb_group'].value_counts().plot.pie(autopct='%1.1f%%')
plt.ylabel('')
plt.show()

# 22
plt.pie(budget, labels=labels, wedgeprops=dict(width=0.4))
plt.text(0, 0, 'Бюджет', ha='center', va='center', fontsize=12)
plt.show()

# 23
data_box = [rng.normal(50, std, 100) for std in (5, 10, 20)]
plt.boxplot(data_box)
plt.xticks([1, 2, 3], ['Группа 1', 'Группа 2', 'Группа 3'])
plt.show()

# 24
plt.boxplot([df['broadband'].dropna(), df['mobile'].dropna()], labels=['Broadband', 'Mobile'])
plt.show()

Q1 = df['broadband'].quantile(0.25)
Q3 = df['broadband'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['broadband'] < (Q1 - 1.5 * IQR)) | (df['broadband'] > (Q3 + 1.5 * IQR))]['country'].tolist()
print("Выбросы broadband:", outliers)

# 25
df['mob_q'] = pd.qcut(df['mobile'], 4)
df.boxplot(column='broadband', by='mob_q')
plt.suptitle('')
plt.show()

# 26
x3d = np.linspace(-5, 5, 50)
y3d = np.linspace(-5, 5, 50)
X, Y = np.meshgrid(x3d, y3d)
Z = np.sin(np.sqrt(X**2 + Y**2))
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.plot_surface(X, Y, Z, cmap='viridis')
plt.show()

# 27
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(rng.random(100), rng.random(100), rng.random(100))
plt.show()

# 28
fig = plt.figure(figsize=(10, 4))
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X, Y, Z, cmap='plasma')
ax1.view_init(elev=30, azim=45)
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X, Y, Z, cmap='coolwarm')
ax2.view_init(elev=60, azim=0)
plt.show()

# 29
mat = rng.random((10, 10))
fig, axes = plt.subplots(1, 3, figsize=(12, 3))
for ax, cm in zip(axes, ['viridis', 'plasma', 'coolwarm']):
    im = ax.imshow(mat, cmap=cm)
    fig.colorbar(im, ax=ax)
plt.show()

# 30
mult_table = np.outer(np.arange(1, 10), np.arange(1, 10))
fig, ax = plt.subplots()
im = ax.imshow(mult_table, cmap='YlGn')
for i in range(9):
    for j in range(9):
        ax.text(j, i, mult_table[i, j], ha='center', va='center', color='black')
plt.show()

# 31
df['ratio'] = df['mobile'] / df['broadband']
df['log_broadband'] = np.log(df['broadband'] + 1)
sns.heatmap(df.corr(numeric_only=True), annot=True, vmin=-1, vmax=1, cmap='coolwarm')
plt.show()

# 32
df2 = pd.DataFrame(rng.normal(size=(200, 4)), columns=list('ABCD'))
df2['D'] = df2['A'] * 0.8 + rng.normal(size=200) * 0.2
sns.heatmap(df2.corr(), annot=True)
plt.show()

# 33
fig, axs = plt.subplots(2, 2)
axs[0,0].plot([1,2,3])
axs[0,1].scatter([1,2],[2,1])
axs[1,0].hist([1,1,2,3])
axs[1,1].bar(['A','B'], [1,2])
fig.suptitle('4 Графика')
fig.tight_layout()
plt.show()

# 34
fig, ax1 = plt.subplots()
ax1.plot(range(1,13), rng.uniform(-10, 30, 12), color='red', label='Температура')
ax2 = ax1.twinx()
ax2.bar(range(1,13), rng.uniform(0, 100, 12), alpha=0.3, label='Осадки')
plt.show()

# 35
x_func = np.linspace(-5, 5, 100)
y_func = -x_func**2 + 10
plt.plot(x_func, y_func)
plt.annotate('Максимум', xy=(0, 10), xytext=(2, 8), arrowprops=dict(facecolor='black', shrink=0.05))
plt.show()

# 36
x_style = np.linspace(0, 10, 100)
y_style = np.sin(x_style)
styles = ['default', 'ggplot', 'seaborn-v0_8-darkgrid'] 
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, style in zip(axes, styles):
    with plt.style.context(style):
        ax.plot(x_style, y_style)
        ax.set_title(style)
plt.show()

# 37
top10 = df.nlargest(10, 'broadband')
fig, ax1 = plt.subplots()
ax1.bar(top10['country'], top10['broadband'], color='blue')
ax2 = ax1.twinx()
ax2.plot(top10['country'], top10['mobile'], color='red', marker='o')
plt.show()

# 38
months_38 = np.arange(1, 13)
sales = rng.integers(100, 300, 12)
fig, ax = plt.subplots(figsize=(10, 5))
ax.bar(months_38, sales, color='lightgreen', alpha=0.6, label='Продажи')
z_trend_38 = np.polyfit(months_38, sales, 1)
p_38 = np.poly1d(z_trend_38)
ax.plot(months_38, p_38(months_38), color='blue', linewidth=2, label='Тренд')
peaks = sales > 200
ax.scatter(months_38[peaks], sales[peaks], color='red', zorder=3, s=80, label='Пики > 200')
ax.set_xticks(months_38)
ax.legend()
plt.title('Продажи по месяцам')
plt.show()

# 39
top15_bb = df.nlargest(15, 'broadband')
mean_bb_all = df['broadband'].mean()
fig, ax = plt.subplots(figsize=(12, 6))
ax.bar(top15_bb['country'], top15_bb['broadband'], color='skyblue')
ax.axhline(mean_bb_all, color='red', linestyle='--', linewidth=2, label='Среднее')
plt.xticks(rotation=45, ha='right')
plt.legend()
plt.tight_layout()
plt.show()

# 40-41
fig, ax = plt.subplots()
ax.plot([1,2,3])
fig.savefig('test100.png', dpi=100)
fig.savefig('test300.png', dpi=300)
fig.savefig('test.pdf')
fig.savefig('test.svg')
fig.savefig('test.jpg', pil_kwargs={'quality': 90})
plt.close(fig)
print(f"Размер PNG 100 dpi: {os.path.getsize('test100.png')} байт")
print(f"Размер PNG 300 dpi: {os.path.getsize('test300.png')} байт")

# 42
with PdfPages('all_plots.pdf') as pdf:
    fig, ax = plt.subplots()
    ax.hist(df['broadband'], bins=20)
    pdf.savefig(fig)
    plt.close(fig)
    
    fig, ax = plt.subplots()
    ax.barh(top10['country'], top10['broadband'])
    pdf.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.boxplot([df['broadband'].dropna(), df['mobile'].dropna()], labels=['Broadband', 'Mobile'])
    pdf.savefig(fig)
    plt.close(fig)

# 43
fig, ax = plt.subplots()
ax.plot([1,2,3])
fig.savefig('test_trans.png', transparent=True, bbox_inches='tight')
plt.close(fig)

# 44
N = 1_000_000
x_big = rng.normal(0, 1, N)
y_big = rng.normal(0, 1, N)

t0 = time.perf_counter()
plt.figure()
plt.scatter(x_big, y_big, s=1)
plt.close()
print(f"Время scatter: {time.perf_counter() - t0:.3f}с")

t0 = time.perf_counter()
plt.figure()
plt.plot(x_big, y_big, "o", markersize=1)
plt.close()
print(f"Время plot: {time.perf_counter() - t0:.3f}с")

# 45
plt.figure()
plt.scatter(x_big, y_big, s=1, alpha=0.1)
plt.title('Плотность с alpha=0.1')
plt.show()

# 46
t0 = time.perf_counter()
plt.figure()
plt.scatter(x_big[::100], y_big[::100], s=1)
plt.title('Каждая 100-я точка')
plt.show()
print(f"Время прореживания: {time.perf_counter() - t0:.3f}с")

# 47
plt.figure()
plt.hexbin(x_big, y_big, gridsize=50, cmap='inferno')
plt.colorbar()
plt.title('Hexbin')
plt.show()

# 48
print(df.info())
print(df.describe())
print(df.isna().sum())

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
ax1.hist(mobile_raw.dropna(), bins=20)
ax1.set_title('До заполнения')
ax2.hist(df['mobile'], bins=20)
ax2.set_title('После заполнения медианой')
plt.show()

# 49
fast_mobile = df[df['ratio'] > 1].sort_values('ratio', ascending=False)
plt.figure(figsize=(10, 6))
plt.barh(fast_mobile['country'], fast_mobile['ratio'], color='coral')
plt.title('Страны, где мобильный интернет быстрее проводного')
plt.tight_layout()
plt.show()

# 50
fig_dash, axs = plt.subplots(2, 2, figsize=(14, 10))
fig_dash.suptitle('Мини-дашборд')

axs[0, 0].hist(df['broadband'], bins=20)
axs[0, 0].set_title('Broadband hist')

axs[0, 1].scatter(df['broadband'], df['mobile'])
z_dash = np.polyfit(df['broadband'], df['mobile'], 1)
p_dash = np.poly1d(z_dash)
axs[0, 1].plot(df['broadband'], p_dash(df['broadband']), "r--")
axs[0, 1].set_title('Scatter + Trend')

axs[1, 0].boxplot([df['broadband'], df['mobile']], labels=['Broadband', 'Mobile'])
axs[1, 0].set_title('Boxplot')

top5_dash = df.nlargest(5, 'broadband').sort_values('broadband')
axs[1, 1].barh(top5_dash['country'], top5_dash['broadband'])
axs[1, 1].set_title('Top-5 Broadband')

plt.tight_layout()
fig_dash.savefig('dashboard.pdf')
fig_dash.savefig('dashboard.png', dpi=300)
plt.show()
