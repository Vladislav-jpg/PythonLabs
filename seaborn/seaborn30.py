import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
sns.set_theme()

tips = pd.read_csv('tips.csv')
penguins = pd.read_csv('penguins.csv')
titanic = pd.read_csv('titanic.csv')
iris = pd.read_csv('iris.csv')
flights = pd.read_csv('flights.csv')
mpg = pd.read_csv('mpg.csv')
fmri = pd.read_csv('fmri.csv')
tips['day'] = pd.Categorical(tips['day'], ['Thur', 'Fri', 'Sat', 'Sun'])

# Задание 1
print(tips.head())
print('Размер таблицы:', tips.shape)
print()
tips.info()
print()
print(tips.describe())
print()
print('Пропуски по столбцам:')
print(tips.isna().sum())

# Задание 2
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
for ax, b in zip(axes, [10, 30]):
    sns.histplot(data=tips, x='total_bill', bins=b, ax=ax)
    ax.set(title=f'Распределение суммы счёта (bins={b})', xlabel='Сумма счёта, $', ylabel='Количество счетов')
plt.tight_layout()
plt.show()

# Задание 3
ax = sns.histplot(data=penguins, x='body_mass_g', kde=True, stat='density')
ax.set(title='Распределение массы тела пингвинов', xlabel='Масса тела, г', ylabel='Плотность')
plt.show()

# Задание 4
ax = sns.countplot(data=titanic, x='class', order=['First', 'Second', 'Third'])
ax.set(title='Количество пассажиров «Титаника» по классам', xlabel='Класс каюты', ylabel='Количество пассажиров')
plt.show()

# Задание 5
ax = sns.countplot(data=titanic, x='survived', hue='sex')
ax.set(title='Выжившие и погибшие по полу (0 - погиб, 1 - выжил)', xlabel='Выжил', ylabel='Количество пассажиров')
plt.show()

# Задание 6
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
sns.barplot(data=tips, x='day', y='tip', errorbar=None, ax=axes[0])
axes[0].set(title='Средние чаевые по дням (без интервала)', xlabel='День недели', ylabel='Средние чаевые, $')
sns.barplot(data=tips, x='day', y='tip', ax=axes[1])
axes[1].set(title='Средние чаевые по дням (95% ДИ)', xlabel='День недели', ylabel='')
plt.tight_layout()
plt.show()

# Задание 7
ax = sns.boxplot(data=penguins, x='species', y='flipper_length_mm', hue='species', legend=False)
ax.set(title='Длина плавника по видам пингвинов', xlabel='Вид', ylabel='Длина плавника, мм')
plt.show()
print(penguins.groupby('species')['flipper_length_mm'].median())

# Задание 8
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
sns.violinplot(data=iris, x='species', y='petal_length', hue='species', legend=False, ax=axes[0])
axes[0].set(title='Скрипичная диаграмма длины лепестка', xlabel='Вид', ylabel='Длина лепестка, см')
sns.boxplot(data=iris, x='species', y='petal_length', hue='species', legend=False, ax=axes[1])
axes[1].set(title='Boxplot длины лепестка', xlabel='Вид', ylabel='')
plt.tight_layout()
plt.show()

# Задание 9
ax = sns.scatterplot(data=tips, x='total_bill', y='tip')
ax.set(title='Сумма счёта и чаевые', xlabel='Сумма счёта, $', ylabel='Чаевые, $')
plt.show()
print('Корреляция:', round(tips['total_bill'].corr(tips['tip']), 2))

# Задание 10
p = penguins.dropna()
ax = sns.scatterplot(data=p, x='bill_length_mm', y='bill_depth_mm', hue='species', style='sex')
ax.set(title='Длина и глубина клюва пингвинов', xlabel='Длина клюва, мм', ylabel='Глубина клюва, мм')
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
plt.show()

# Задание 11
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
sns.lineplot(data=flights, x='year', y='passengers', ax=axes[0])
axes[0].set(title='Пассажиры по годам (с интервалом)', xlabel='Год', ylabel='Число пассажиров, тыс.')
sns.lineplot(data=flights, x='year', y='passengers', errorbar=None, ax=axes[1])
axes[1].set(title='Пассажиры по годам (без интервала)', xlabel='Год', ylabel='')
plt.tight_layout()
plt.show()

# Задание 12
for style in ['darkgrid', 'whitegrid', 'ticks']:
    with sns.axes_style(style):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
        for ax, pal in zip(axes, ['Set2', 'pastel']):
            sns.boxplot(data=tips, x='day', y='total_bill', hue='day', palette=pal, dodge=False, legend=False, ax=ax)
            ax.set(title=f'style={style}, palette={pal}', xlabel='День недели', ylabel='Сумма счёта, $')
        plt.tight_layout()
        plt.show()

# Задание 13
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=penguins, x='island', hue='island', legend=False)
ax.set_title('Количество пингвинов по островам')
ax.set_xlabel('Остров')
ax.set_ylabel('Количество пингвинов')
plt.savefig('penguins_islands.png', dpi=300, bbox_inches='tight')
plt.show()

# Задание 14
corr = iris.corr(numeric_only=True)
ax = sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0)
ax.set_title('Корреляции признаков ирисов')
plt.xticks(rotation=30)
plt.show()

# Задание 15
ax = sns.stripplot(data=tips, x='day', y='total_bill', hue='sex', dodge=True)
ax.set(title='Суммы счетов по дням и полу', xlabel='День недели', ylabel='Сумма счёта, $')
plt.show()

# Задание 16
fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=True)
sns.histplot(data=penguins, x='flipper_length_mm', hue='species', multiple='layer', element='step', ax=axes[0])
axes[0].set(title='Наложение (layer, step)', xlabel='Длина плавника, мм', ylabel='Количество')
sns.histplot(data=penguins, x='flipper_length_mm', hue='species', multiple='stack', ax=axes[1])
axes[1].set(title='Накопление (stack)', xlabel='Длина плавника, мм', ylabel='')
plt.tight_layout()
plt.show()

# Задание 17
ax = sns.kdeplot(data=penguins, x='bill_length_mm', y='bill_depth_mm', hue='species', fill=True, alpha=0.5)
ax.set(title='Двумерная плотность размеров клюва', xlabel='Длина клюва, мм', ylabel='Глубина клюва, мм')
plt.show()

# Задание 18
ax = sns.ecdfplot(data=tips, x='total_bill', hue='time')
ax.axvline(20, color='gray', linestyle='--')
ax.set(title='ЭФР суммы счёта: обед и ужин', xlabel='Сумма счёта, $', ylabel='Доля счетов')
plt.show()
print(tips.groupby('time', observed=True)['total_bill'].apply(lambda s: (s <= 20).mean()).round(2))

# Задание 19
g = sns.catplot(data=tips, x='day', y='total_bill', hue='sex', col='time', kind='box', height=4, aspect=1)
g.set_titles('{col_name}')
g.set_axis_labels('День недели', 'Сумма счёта, $')
g.figure.suptitle('Суммы счетов по дням, полу и времени', y=1.03)
plt.show()

# Задание 20
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=penguins, x='species', y='body_mass_g', hue='species', legend=False, showfliers=False, ax=ax)
sns.stripplot(data=penguins, x='species', y='body_mass_g', color='black', size=3, alpha=0.5, ax=ax)
ax.set(title='Масса тела пингвинов по видам', xlabel='Вид', ylabel='Масса тела, г')
plt.show()

# Задание 21
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

ax = axes[0]
sns.barplot(data=titanic, x='class', y='survived', hue='sex', order=['First', 'Second', 'Third'], ax=ax)
for c in ax.containers:
    ax.bar_label(c, fmt='%.2f')
ax.set(title='Доля выживших по классам и полу', xlabel='Класс', ylabel='Доля выживших')

ax = axes[1]
sns.barplot(data=titanic, x='class', y='survived', hue='sex', order=['First', 'Second', 'Third'],
            estimator='sum', errorbar=None, ax=ax)
for c in ax.containers:
    ax.bar_label(c, fmt='%.0f')
ax.set(title='Число выживших по классам и полу', xlabel='Класс', ylabel='Число выживших')
plt.tight_layout()
plt.show()

# Задание 22
m = mpg.dropna()
ax = sns.scatterplot(data=m, x='horsepower', y='mpg', hue='origin', size='weight', sizes=(20, 200), alpha=0.6)
ax.set(title='Расход топлива, мощность, вес и регион', xlabel='Мощность, л.с.', ylabel='Расход, миль на галлон')
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
plt.show()

# Задание 23
g = sns.FacetGrid(penguins, col='species', row='sex', height=3)
g.map_dataframe(sns.histplot, x='body_mass_g')
g.set_axis_labels('Масса тела, г', 'Количество')
g.set_titles(col_template='{col_name}', row_template='{row_name}')
g.figure.suptitle('Масса тела по видам и полу', y=1.02)
plt.show()

# Задание 24
fig, axes = plt.subplots(1, 3, figsize=(18, 4.8), sharey=True)
sns.lineplot(data=fmri, x='timepoint', y='signal', hue='event', style='region', ax=axes[0])
axes[0].set(title='По умолчанию (95% ДИ)', xlabel='Момент времени', ylabel='Сигнал')
sns.lineplot(data=fmri, x='timepoint', y='signal', hue='event', style='region', errorbar='sd', ax=axes[1])
axes[1].set(title="errorbar='sd'", xlabel='Момент времени', ylabel='')
sns.lineplot(data=fmri, x='timepoint', y='signal', hue='event', style='region', errorbar=('ci', 95), ax=axes[2])
axes[2].set(title="errorbar=('ci', 95)", xlabel='Момент времени', ylabel='')
plt.tight_layout()
plt.show()

g = sns.relplot(data=fmri, x='timepoint', y='signal', hue='event', kind='line', col='region', height=4)
g.set_axis_labels('Момент времени', 'Сигнал')
g.set_titles('Область: {col_name}')
plt.show()

# Задание 25
g = sns.lmplot(data=tips, x='total_bill', y='tip', hue='smoker', col='time', height=4.5)
g.set_axis_labels('Сумма счёта, $', 'Чаевые, $')
g.set_titles('{col_name}')
plt.show()

# Задание 26
m = mpg.dropna()
fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
sns.regplot(data=m, x='horsepower', y='mpg', order=1, scatter_kws={'alpha': 0.5}, ax=axes[0])
axes[0].set(title='Линейная регрессия (order=1)', xlabel='Мощность, л.с.', ylabel='Расход, миль на галлон')
sns.regplot(data=m, x='horsepower', y='mpg', order=2, scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=axes[1])
axes[1].set(title='Полиномиальная регрессия (order=2)', xlabel='Мощность, л.с.', ylabel='Расход, миль на галлон')
sns.residplot(data=m, x='horsepower', y='mpg', scatter_kws={'alpha': 0.5}, ax=axes[2])
axes[2].set(title='Остатки линейной модели', xlabel='Мощность, л.с.', ylabel='Остаток')
plt.tight_layout()
plt.show()

# Задание 27
pivot = flights.pivot(index='month', columns='year', values='passengers').loc[flights['month'].unique()]
ax = sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5, cbar_kws={'label': 'Число пассажиров, тыс.'})
ax.set(title='Пассажиры по месяцам и годам', xlabel='Год', ylabel='Месяц')
plt.gcf().set_size_inches(12, 6)
plt.show()

# Задание 28
cols = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
g = sns.pairplot(penguins, vars=cols, hue='species', diag_kind='kde', corner=True)
g.figure.suptitle('Попарные связи признаков пингвинов', y=1.02)
plt.show()

# Задание 29
g1 = sns.jointplot(data=penguins, x='flipper_length_mm', y='body_mass_g', kind='scatter', hue='species')
g1.set_axis_labels('Длина плавника, мм', 'Масса тела, г')
g1.figure.suptitle('Scatter с разбивкой по видам', y=1.02)
g2 = sns.jointplot(data=penguins, x='flipper_length_mm', y='body_mass_g', kind='hex')
g2.set_axis_labels('Длина плавника, мм', 'Масса тела, г')
g2.figure.suptitle('Hex: плотность наблюдений', y=1.02)
g3 = sns.jointplot(data=penguins, x='flipper_length_mm', y='body_mass_g', kind='reg')
g3.set_axis_labels('Длина плавника, мм', 'Масса тела, г')
g3.figure.suptitle('Reg: линейная регрессия', y=1.02)
plt.show()

# Задание 30
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
sns.countplot(data=penguins, x='island', hue='species', ax=axes[0, 0])
axes[0, 0].set(title='Виды по островам', xlabel='Остров', ylabel='Количество')
sns.boxplot(data=penguins, x='species', y='body_mass_g', hue='species', legend=False, ax=axes[0, 1])
axes[0, 1].set(title='Масса тела по видам', xlabel='Вид', ylabel='Масса тела, г')
sns.scatterplot(data=penguins, x='flipper_length_mm', y='body_mass_g', hue='sex', ax=axes[1, 0])
axes[1, 0].set(title='Длина плавника и масса тела', xlabel='Длина плавника, мм', ylabel='Масса тела, г')
sns.heatmap(penguins.corr(numeric_only=True), annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Корреляции признаков')
axes[1, 1].tick_params(axis='x', rotation=30)
fig.suptitle('Панель анализа данных о пингвинах', fontsize=16)
plt.tight_layout()
plt.savefig('penguins_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
