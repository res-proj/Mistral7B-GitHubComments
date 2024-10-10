import pandas as pd
from statsmodels.stats.contingency_tables import mcnemar


strong_df = pd.read_csv('', encoding='latin-1')
weak_df = pd.read_csv('', encoding='latin-1')

# Remove rows where the "comment" column is blank or where "connection" column contains "self"
strong_df = strong_df[strong_df['comment'].notna() & (strong_df['comment'] != '') & (strong_df['connection'] != 'self')]
weak_df = weak_df[weak_df['comment'].notna() & (weak_df['comment'] != '') & (weak_df['connection'] != 'self')]


strong_pivot = strong_df.pivot_table(index='commentor', columns='Theme', aggfunc='size', fill_value=0)
weak_pivot = weak_df.pivot_table(index='commentor', columns='Theme', aggfunc='size', fill_value=0)


results = {}


themes = ['Conflict', 'Improvement', 'Inquiry', 'Interdependence', 'Project Workflow','Miscellaneous']

for theme in themes:

    strong_counts = strong_pivot[theme] if theme in strong_pivot else pd.Series(0, index=strong_pivot.index)
    weak_counts = weak_pivot[theme] if theme in weak_pivot else pd.Series(0, index=weak_pivot.index)


    combined_df = pd.DataFrame({'strong': strong_counts, 'weak': weak_counts}).fillna(0).astype(int)


    both = ((combined_df['strong'] > 0) & (combined_df['weak'] > 0)).sum()
    strong_only = ((combined_df['strong'] > 0) & (combined_df['weak'] == 0)).sum()
    weak_only = ((combined_df['strong'] == 0) & (combined_df['weak'] > 0)).sum()
    neither = ((combined_df['strong'] == 0) & (combined_df['weak'] == 0)).sum()  

    contingency_table = [[both, strong_only], [weak_only, neither]]


    result = mcnemar(contingency_table, exact=True)
    results[theme] = result.pvalue


for theme, pvalue in results.items():
    print(f"Theme: {theme}, McNemar's Test p-value: {pvalue:.4f}")