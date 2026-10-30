import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.backends.backend_pdf import PdfPages

df = pd.read_csv("Internet Speed 2022.csv")
df['mobile'] = df['mobile'].fillna(df['mobile'].median())
df['ratio'] = df['mobile'] / df['broadband']
rng = np.random.default_rng(42)

# 51
x = np.linspace(0, 4*np.pi, 200)
y = np.sin(x)
plt.plot(x, y)
plt.fill_between(x, y, 0, where=(y > 0), color='green', alpha=0.5)
plt.fill_between(x, y, 0, where=(y < 0), color='red', alpha=0.5)
plt.show()

# 52
x_err = np.arange(10)
y_mean = rng.uniform(5, 10, 10)
y_err = rng.uniform(0.5, 1.5, 10)
plt.errorbar(x_err, y_mean, yerr=y_err, fmt='-o')
plt.show()

# 53
months = np.arange(1, 13)
rev = [rng.uniform(10, 30, 12) for _ in range(3)]
plt.stackplot(months, rev, labels=['A', 'B', 'C'])
plt.legend()
plt.show()

# 54
fig = plt.figure(figsize=(10, 6))
gs = fig.add_gridspec(2, 2)
ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[1, 1])
ax1.plot(months, rev[0], color='navy')
ax1.set_title('Широкий график сверху')
ax2.hist(rng.normal(0, 1, 100), color='seagreen')
ax2.set_title('Маленький слева')
ax3.scatter(rng.random(20), rng.random(20), color='crimson')
ax3.set_title('Маленький справа')
plt.tight_layout()
plt.show()

# 55
fig, ax = plt.subplots()
ax.plot(x_err, y_mean)
ax.axvspan(2, 4, color='yellow', alpha=0.3)
ax.axhline(7, color='red', linestyle='--')
ax.text(3, 7.5, 'Интервал')
ax.text(0, 7.2, 'Порог')
plt.show()

# 56
def setup_style():
    plt.rcParams.update({'font.size': 14, 'lines.linewidth': 3})
setup_style()

# 57
top10_pandas = df.nlargest(10, 'broadband').set_index('country')
top10_pandas['broadband'].plot.barh()
plt.title('Топ-10 Broadband (Pandas)')
plt.show()

# 58
df.plot.scatter(x='broadband', y='mobile', c='ratio', colormap='viridis')
plt.title('Broadband vs Mobile (цвет по ratio)')
plt.show()

# 59
quants = df['broadband'].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
plt.hist(df['broadband'], bins=20, color='lightgray', edgecolor='black')
colors = ['r', 'g', 'b', 'm', 'c']
for q, color in zip(quants, colors):
    plt.axvline(q, color=color, linewidth=2, label=f'p={q:.1f}')
plt.legend()
plt.show()

# 60
real_continent_dict = {
    'Monaco': 'Europe', 'Singapore': 'Asia', 'Hong Kong': 'Asia', 'Romania': 'Europe', 'Switzerland': 'Europe',
    'Denmark': 'Europe', 'Thailand': 'Asia', 'Chile': 'South America', 'France': 'Europe', 'South Korea': 'Asia',
    'United States': 'North America', 'China': 'Asia', 'Canada': 'North America', 'Japan': 'Asia',
    'United Arab Emirates': 'Asia', 'Sweden': 'Europe', 'New Zealand': 'Oceania', 'Norway': 'Europe',
    'Israel': 'Asia', 'Germany': 'Europe', 'Russia': 'Europe', 'Brazil': 'South America', 'Australia': 'Oceania',
    'South Africa': 'Africa', 'Nigeria': 'Africa', 'Kenya': 'Africa', 'Egypt': 'Africa', 'India': 'Asia'
}
df['continent'] = df['country'].map(real_continent_dict)
df_continents = df.dropna(subset=['continent'])
mean_speed = df_continents.groupby('continent')['broadband'].mean()
mean_speed.plot.bar(color='slateblue')
plt.title('Средняя скорость по континентам')
plt.xticks(rotation=45)
plt.show()

# 61
df_continents.boxplot(column='broadband', by='continent', figsize=(8, 6))
plt.title('Распределение скорости broadband')
plt.suptitle('')
plt.xticks(rotation=45)
plt.show()

# 62
df_sort = df.sort_values('broadband', ascending=False)
cum_pct = df_sort['broadband'].cumsum() / df_sort['broadband'].sum() * 100
plt.plot(range(len(cum_pct)), cum_pct, color='navy', linewidth=2)
plt.axhline(80, color='r', linestyle='--')
plt.title('Кривая Парето (Broadband)')
plt.show()

# 63
sns.histplot(data=df, x='broadband', kde=True)
plt.show()

# 64
sns.regplot(data=df, x='broadband', y='mobile', scatter_kws={'alpha':0.5})
plt.show()

# 65
sns.pairplot(df[['broadband', 'mobile']])
plt.show()

# 66
df['tier'] = pd.qcut(df['broadband'], 3, labels=['Low', 'Med', 'High'])
sns.violinplot(data=df, x='tier', y='mobile')
plt.show()

# 67
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
ax1.scatter(df['broadband'], df['mobile'])
ax1.set_title('Обычный масштаб')
ax2.scatter(df['broadband'], df['mobile'])
ax2.set_xscale('log')
ax2.set_title('Логарифмический масштаб (Ось X)')
plt.show()

# 69
def plot_top(dataframe, column, n=10, title=""):
    d = dataframe.nlargest(n, column).sort_values(column)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(d['country'], d[column], color='teal')
    ax.set_title(title)
    return fig

fig_top10 = plot_top(df, 'broadband', title='Top 10 Broadband')
plt.show()

# 70
with PdfPages('mini_report.pdf') as pdf:
    pdf.savefig(fig_top10)
