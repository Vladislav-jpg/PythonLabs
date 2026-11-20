import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

sns.set_theme()

penguins = pd.read_csv('penguins.csv')
iris     = pd.read_csv('iris.csv')
titanic  = pd.read_csv('titanic.csv')
flights  = pd.read_csv('flights.csv')
mpg      = pd.read_csv('mpg.csv')
fmri     = pd.read_csv('fmri.csv')

DAYS = ['Thur', 'Fri', 'Sat', 'Sun']

#1
tips = pd.read_csv('tips.csv')

print('Первые 5 строк: ')
print(tips.head())
print('Размер таблицы (строки, столбцы): ', tips.shape)
print()
tips.info()
print()
print(tips.describe())

num_cols = tips.select_dtypes(include='number').columns.tolist()
cat_cols = [c for c in tips.columns if c not in num_cols]
print('Числовые столбцы: ', num_cols)
print('Категориальные столбцы: ', cat_cols)
print()
print('Пропуски по столбцам: ')
print(tips.isna().sum())

#2
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
for ax, b in zip(axes, [10, 30]):
    sns.histplot(data=tips, x='total_bill', bins=b, ax=ax)
    ax.set_title(f'Распределение суммы счёта (bins={b})')
    ax.set_xlabel('Сумма счёта')
    ax.set_ylabel('Количество счетов')
fig.tight_layout()
plt.show()

#3
fig, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=penguins, x='body_mass_g', kde=True, stat='density', ax=ax)
ax.set_title('Распределение массы тела пингвинов')
ax.set_xlabel('Масса тела, г')
ax.set_ylabel('Плотность')
plt.show()

print('Асимметрия (skew):', round(penguins['body_mass_g'].skew(), 2))

#4
fig, ax = plt.subplots(figsize=(7, 5))
sns.countplot(data=titanic, x='class', order=['First', 'Second', 'Third'], ax=ax)
ax.set_title('Количество пассажиров по классам')
ax.set_xlabel('Класс каюты')
ax.set_ylabel('Количество пассажиров')
plt.show()

print(titanic['class'].value_counts())

#5
fig, ax = plt.subplots(figsize=(7, 5))
sns.countplot(data=titanic, x='survived', hue='sex', ax=ax)
ax.set_title('Выживаемость пассажиров Титаника по полу')
ax.set_xlabel('Исход')
ax.set_ylabel('Количество пассажиров')
ax.set_xticks([0, 1])
ax.set_xticklabels(['Погиб', 'Выжил'])
sns.move_legend(ax, 'upper right', title='Пол')
plt.show()

print(titanic.groupby('sex')['survived'].agg(['sum', 'count', 'mean']).round(3))

#6
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

sns.barplot(data=tips, x='day', y='tip', order=DAYS, errorbar=None, ax=axes[0])
axes[0].set_title('Средние чаевые по дням (без доверительного интервала)')

sns.barplot(data=tips, x='day', y='tip', order=DAYS, errorbar=('ci', 95), ax=axes[1])
axes[1].set_title('Средние чаевые по дням (с 95% ДИ)')

for ax in axes:
    ax.set_xlabel('День недели')
    ax.set_ylabel('Средние чаевые')
fig.tight_layout()
plt.show()

print(tips.groupby('day')['tip'].mean().reindex(DAYS).round(2))

#7
fig, ax = plt.subplots(figsize=(7, 5))
sns.boxplot(data=penguins, x='species', y='flipper_length_mm', hue='species', legend=False, ax=ax)
ax.set_title('Длина плавника по видам пингвинов')
ax.set_xlabel('Вид')
ax.set_ylabel('Длина плавника, мм')
plt.show()

print(penguins.groupby('species')['flipper_length_mm'].median())

#8
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

sns.violinplot(data=iris, x='species', y='petal_length', hue='species', legend=False, ax=axes[0])
axes[0].set_title('Скрипичная диаграмма длины лепестка')

sns.boxplot(data=iris, x='species', y='petal_length', hue='species', legend=False, ax=axes[1])
axes[1].set_title('Boxplot длины лепестка')

for ax in axes:
    ax.set_xlabel('Вид ириса')
    ax.set_ylabel('Длина лепестка, см')
fig.tight_layout()
plt.show()

#9
fig, ax = plt.subplots(figsize=(7, 5))
sns.scatterplot(data=tips, x='total_bill', y='tip', ax=ax)
ax.set_title('Связь суммы счёта и чаевых')
ax.set_xlabel('Сумма счёта')
ax.set_ylabel('Чаевые')
plt.show()

print('Коэффициент корреляции:', round(tips['total_bill'].corr(tips['tip']), 2))

#10
peng = penguins.dropna()

fig, ax = plt.subplots(figsize=(8, 6))
sns.scatterplot(data=peng, x='bill_length_mm', y='bill_depth_mm',
                hue='species', style='sex', s=60, ax=ax)
ax.set_title('Длина и глубина клюва пингвинов')
ax.set_xlabel('Длина клюва, мм')
ax.set_ylabel('Глубина клюва, мм')
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
plt.show()

#11
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)

sns.lineplot(data=flights, x='year', y='passengers', ax=axes[0])
axes[0].set_title('Пассажиры по годам (с доверительной полосой)')

sns.lineplot(data=flights, x='year', y='passengers', errorbar=None, ax=axes[1])
axes[1].set_title('Пассажиры по годам (errorbar=None)')

for ax in axes:
    ax.set_xlabel('Год')
    ax.set_ylabel('Среднее число пассажиров в месяц')
fig.tight_layout()
plt.show()

#12
styles   = ['darkgrid', 'whitegrid', 'ticks']
palettes = ['Set2', 'pastel']

fig = plt.figure(figsize=(15, 8))
for i, pal in enumerate(palettes):
    for j, st in enumerate(styles):
        with sns.axes_style(st):
            ax = fig.add_subplot(2, 3, i * 3 + j + 1)
        sns.boxplot(data=tips, x='day', y='total_bill', order=DAYS,
                    hue='day', hue_order=DAYS, palette=pal, legend=False, ax=ax)
        ax.set_title(f'style={st}, palette={pal}')
        ax.set_xlabel('День недели')
        ax.set_ylabel('Сумма счёта')
fig.tight_layout()
plt.show()

#13
plt.figure(figsize=(8, 5))
ax = sns.countplot(data=penguins, x='island', hue='island', legend=False)
ax.set_title('Количество пингвинов по островам')
ax.set_xlabel('Остров')
ax.set_ylabel('Количество пингвинов')
plt.savefig('penguins_islands.png', dpi=300, bbox_inches='tight')
plt.show()

print(penguins['island'].value_counts())

#14
corr = iris.corr(numeric_only=True)

fig, ax = plt.subplots(figsize=(7, 5.5))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, ax=ax)
ax.set_title('Корреляции признаков ирисов')
ax.set_xlabel('Признак')
ax.set_ylabel('Признак')
plt.show()

pairs = corr.where(~np.eye(len(corr), dtype=bool)).abs().unstack().sort_values(ascending=False)
print('Самая сильная пара:', pairs.index[0], round(corr.loc[pairs.index[0]], 2))

#15
fig, ax = plt.subplots(figsize=(8, 5))
sns.stripplot(data=tips, x='day', y='total_bill', hue='sex', order=DAYS, dodge=True, ax=ax)
ax.set_title('Суммы счетов по дням недели и полу')
ax.set_xlabel('День недели')
ax.set_ylabel('Сумма счёта')
sns.move_legend(ax, 'upper left', title='Пол')
plt.show()

print(tips['day'].value_counts().reindex(DAYS))

#16
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

sns.histplot(data=penguins, x='flipper_length_mm', hue='species',
             multiple='layer', element='step', ax=axes[0])
axes[0].set_title('Наложение')

sns.histplot(data=penguins, x='flipper_length_mm', hue='species',
             multiple='stack', ax=axes[1])
axes[1].set_title('Накопление')

for ax in axes:
    ax.set_xlabel('Длина плавника, мм')
    ax.set_ylabel('Количество пингвинов')
sns.move_legend(axes[0], 'upper left', title='Вид')
sns.move_legend(axes[1], 'upper left', title='Вид')
fig.tight_layout()
plt.show()

#17
fig, ax = plt.subplots(figsize=(8, 6))
sns.kdeplot(data=penguins, x='bill_length_mm', y='bill_depth_mm',
            hue='species', fill=True, alpha=0.5, ax=ax)
ax.set_title('Двумерная плотность длины и глубины клюва')
ax.set_xlabel('Длина клюва, мм')
ax.set_ylabel('Глубина клюва, мм')
sns.move_legend(ax, 'upper right', title='Вид')
plt.show()

#18
fig, ax = plt.subplots(figsize=(8, 5))
sns.ecdfplot(data=tips, x='total_bill', hue='time', ax=ax)
ax.axvline(20, color='gray', linestyle='--')
ax.set_title('ECDF суммы счёта: обед и ужин')
ax.set_xlabel('Сумма счёта')
ax.set_ylabel('Доля счетов')
sns.move_legend(ax, 'lower right', title='Время')
plt.show()

print('Доля счетов не выше 20 $:')
print(tips.groupby('time')['total_bill'].apply(lambda s: (s <= 20).mean()).round(3))

#19
g = sns.catplot(data=tips, x='day', y='total_bill', hue='sex', col='time',
                kind='box', order=DAYS, height=4, aspect=1)
g.set_titles(col_template='Время: {col_name}')
g.set_axis_labels('День недели', 'Сумма счёта')
sns.move_legend(g, 'center right', title='Пол')
plt.show()

#20
fig, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(data=penguins, x='species', y='body_mass_g', showfliers=False,
            hue='species', legend=False, ax=ax)
sns.stripplot(data=penguins, x='species', y='body_mass_g',
              color='black', size=3, alpha=0.5, ax=ax)
ax.set_title('Масса тела пингвинов по видам')
ax.set_xlabel('Вид')
ax.set_ylabel('Масса тела, г')
plt.show()

#21
order = ['First', 'Second', 'Third']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.barplot(data=titanic, x='class', y='survived', hue='sex', order=order, ax=axes[0])
axes[0].set_title('Доля выживших по классам и полу (среднее)')
axes[0].set_ylabel('Доля выживших')
for c in axes[0].containers:
    axes[0].bar_label(c, fmt='%.2f')

sns.barplot(data=titanic, x='class', y='survived', hue='sex', order=order,
            estimator='sum', errorbar=None, ax=axes[1])
axes[1].set_title('Число выживших по классам и полу (сумма)')
axes[1].set_ylabel('Число выживших')
for c in axes[1].containers:
    axes[1].bar_label(c, fmt='%d')

for ax in axes:
    ax.set_xlabel('Класс каюты')
    sns.move_legend(ax, 'upper right', title='Пол')
fig.tight_layout()
plt.show()

#22
mpg_c = mpg.dropna()

fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(data=mpg_c, x='horsepower', y='mpg', hue='origin', size='weight',
                sizes=(20, 200), alpha=0.6, ax=ax)
ax.set_title('Расход топлива и мощность автомобилей')
ax.set_xlabel('Мощность, л. с.')
ax.set_ylabel('Расход, миль на галлон (mpg)')
sns.move_legend(ax, 'upper left', bbox_to_anchor=(1, 1))
plt.show()

print('Корреляция:', round(mpg_c['horsepower'].corr(mpg_c['mpg']), 2))
print(mpg_c.groupby('origin')[['horsepower', 'mpg', 'weight']].mean().round(1))

#23
g = sns.FacetGrid(penguins, col='species', row='sex', height=3)
g.map_dataframe(sns.histplot, x='body_mass_g')
g.set_axis_labels('Масса тела, г', 'Количество')
g.set_titles(col_template='Вид: {col_name}', row_template='Пол: {row_name}')
g.figure.suptitle('Распределение массы тела по видам и полу', y=1.03)
plt.show()

print(penguins.groupby(['species', 'sex'])['body_mass_g'].mean().round(0))

#24
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

sns.lineplot(data=fmri, x='timepoint', y='signal', hue='event', style='region',
             errorbar='sd', ax=axes[0])
axes[0].set_title('Разброс: errorbar="sd"')

sns.lineplot(data=fmri, x='timepoint', y='signal', hue='event', style='region',
             errorbar=('ci', 95), ax=axes[1])
axes[1].set_title('95% доверительный интервал среднего')

for ax in axes:
    ax.set_xlabel('Момент времени')
    ax.set_ylabel('Сигнал фМРТ')
fig.tight_layout()
plt.show()

g = sns.relplot(data=fmri, x='timepoint', y='signal', hue='event',
                kind='line', col='region', height=4, aspect=1.2)
g.set_titles(col_template='Область: {col_name}')
g.set_axis_labels('Момент времени', 'Сигнал фМРТ')
plt.show()

#25
g = sns.lmplot(data=tips, x='total_bill', y='tip', hue='smoker', col='time',
               height=4.5, aspect=1)
g.set_titles(col_template='Время: {col_name}')
g.set_axis_labels('Сумма счёта', 'Чаевые')
sns.move_legend(g, 'center right', title='Курящий')
plt.show()

for (t, s), grp in tips.groupby(['time', 'smoker']):
    slope = np.polyfit(grp['total_bill'], grp['tip'], 1)[0]
    print(f'{t:7s} smoker={s:3s} n={len(grp):3d} наклон={slope:.3f}')

#26
fig, axes = plt.subplots(1, 3, figsize=(17, 5))

sns.regplot(data=mpg_c, x='horsepower', y='mpg', order=1,
            scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=axes[0])
axes[0].set_title('Линейная регрессия (order=1)')

sns.regplot(data=mpg_c, x='horsepower', y='mpg', order=2,
            scatter_kws={'alpha': 0.5}, line_kws={'color': 'red'}, ax=axes[1])
axes[1].set_title('Полиномиальная регрессия (order=2)')

sns.residplot(data=mpg_c, x='horsepower', y='mpg', order=1,
              scatter_kws={'alpha': 0.5}, ax=axes[2])
axes[2].set_title('Остатки линейной модели')

axes[0].set_xlabel('Мощность, л. с.'); axes[0].set_ylabel('mpg')
axes[1].set_xlabel('Мощность, л. с.'); axes[1].set_ylabel('mpg')
axes[2].set_xlabel('Мощность, л. с.'); axes[2].set_ylabel('Остаток (mpg)')
fig.tight_layout()
plt.show()

for k in (1, 2):
    c = np.polyfit(mpg_c['horsepower'], mpg_c['mpg'], k)
    resid = mpg_c['mpg'] - np.polyval(c, mpg_c['horsepower'])
    print(f'order={k}: R² = {1 - resid.var() / mpg_c["mpg"].var():.3f}')

#27
month_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July',
               'August', 'September', 'October', 'November', 'December']
pivot = flights.pivot(index='month', columns='year', values='passengers').reindex(month_order)

fig, ax = plt.subplots(figsize=(13, 7))
sns.heatmap(pivot, annot=True, fmt='d', cmap='YlOrRd', linewidths=0.5,
            cbar_kws={'label': 'Число пассажиров'}, ax=ax)
ax.set_title('Число авиапассажиров по месяцам и годам')
ax.set_xlabel('Год')
ax.set_ylabel('Месяц')
plt.show()

#28
num = ['bill_length_mm', 'bill_depth_mm', 'flipper_length_mm', 'body_mass_g']
g = sns.pairplot(penguins, vars=num, hue='species', diag_kind='kde', corner=True)
g.figure.suptitle('Попарные диаграммы рассеяния признаков пингвинов', y=1.02)
sns.move_legend(g, 'center right', title='Вид')
plt.show()

#29
g1 = sns.jointplot(data=penguins, x='flipper_length_mm', y='body_mass_g',
                   hue='species', kind='scatter')
g1.set_axis_labels('Длина плавника, мм', 'Масса тела, г')
sns.move_legend(g1.ax_joint, 'lower right', title='Вид')
g1.figure.suptitle('Jointplot: scatter с разбивкой по видам', y=1.02)

g2 = sns.jointplot(data=penguins, x='flipper_length_mm', y='body_mass_g', kind='hex')
g2.set_axis_labels('Длина плавника, мм', 'Масса тела, г')
g2.figure.suptitle('Jointplot: hex (плотность)', y=1.02)

g3 = sns.jointplot(data=penguins, x='flipper_length_mm', y='body_mass_g', kind='reg')
g3.set_axis_labels('Длина плавника, мм', 'Масса тела, г')
g3.figure.suptitle('Jointplot: reg (линейная регрессия)', y=1.02)
plt.show()

print('Корреляция:', round(penguins['flipper_length_mm'].corr(penguins['body_mass_g']), 2))

#30
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

sns.countplot(data=penguins, x='island', hue='species', ax=axes[0, 0])
axes[0, 0].set_title('Виды пингвинов по островам')
axes[0, 0].set_xlabel('Остров')
axes[0, 0].set_ylabel('Количество пингвинов')
sns.move_legend(axes[0, 0], 'upper right', title='Вид')

sns.boxplot(data=penguins, x='species', y='body_mass_g', hue='species',
            legend=False, ax=axes[0, 1])
axes[0, 1].set_title('Масса тела по видам')
axes[0, 1].set_xlabel('Вид')
axes[0, 1].set_ylabel('Масса тела, г')

sns.scatterplot(data=penguins, x='flipper_length_mm', y='body_mass_g',
                hue='sex', ax=axes[1, 0])
axes[1, 0].set_title('Длина плавника и масса тела')
axes[1, 0].set_xlabel('Длина плавника, мм')
axes[1, 0].set_ylabel('Масса тела, г')
sns.move_legend(axes[1, 0], 'upper left', title='Пол')

sns.heatmap(penguins.corr(numeric_only=True), annot=True, fmt='.2f',
            cmap='coolwarm', center=0, ax=axes[1, 1])
axes[1, 1].set_title('Корреляции числовых признаков')
axes[1, 1].set_xlabel('Признак')
axes[1, 1].set_ylabel('Признак')
axes[1, 1].tick_params(axis='x', rotation=30)

fig.suptitle('Итоговая панель:', fontsize=16)
plt.tight_layout()
plt.savefig('penguins_dashboard.png', dpi=300, bbox_inches='tight')
plt.show()
