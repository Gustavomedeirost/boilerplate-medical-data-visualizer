import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# 1
df = pd.read_csv('medical_examination.csv')

# 2
df['overweight'] = (df['weight'] / ((df['height'] / 100) ** 2)).apply(lambda imc: 1 if imc > 25 else 0)

# 3
df['cholesterol'] = df ['cholesterol'].apply(lambda x: 0 if x == 1 else 1)
df['gluc'] = df['gluc'].apply(lambda x: 0 if x == 1 else 1)

# 4
def draw_cat_plot():
    # 5
    df_cat = pd.melt(
        df,
        id_vars=['cardio'],
        value_vars=['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke']
    )

    # Garante todas as combinações possíveis
    df_cat = df_cat.groupby(['cardio', 'variable', 'value'], as_index=False).size()
    df_cat = df_cat.rename(columns={'size': 'total'})

    # Garante que todas as combinações estejam presentes
    all_combinations = pd.MultiIndex.from_product(
        [[0, 1], ['active', 'alco', 'cholesterol', 'gluc', 'overweight', 'smoke'], [0, 1]],
        names=['cardio', 'variable', 'value']
    )
    df_cat = df_cat.set_index(['cardio', 'variable', 'value']).reindex(all_combinations, fill_value=0).reset_index()
    

    # 7
    fig = sns.catplot(x='variable', y='total', hue='value', col='cardio', data=df_cat, kind='bar')
    fig.set_axis_labels("variable", "total")
    fig.set_titles("{col_name} Cardio")


    # 8
    fig = fig.figure 


    # 9
    fig.savefig('catplot.png')
    return fig


# 10
def draw_heat_map():
    # 11
    df_heat = df[
        (df['ap_lo'] <= df['ap_hi']) &
        (df['height'] >= df['height'].quantile(0.025)) &
        (df['height'] <= df['height'].quantile(0.975)) &
        (df['weight'] >= df['weight'].quantile(0.025)) &
        (df['weight'] <= df['weight'].quantile(0.975))
    ]

    # 12
    corr = df_heat.corr()

    # 13
    mask = np.triu(np.ones_like(corr, dtype=bool))

    # 14
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt='.1f', mask=mask, center=0, vmax=.3, square=True, cbar_kws={"shrink": .5})

    # 15
    ax.set_title('Correlation Heatmap')

    # 16
    fig.savefig('heatmap.png')
    return fig
