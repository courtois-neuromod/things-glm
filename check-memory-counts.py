from pathlib import Path

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


def f(row):
    """
    # TODO : Double-check that these are being correctly generated
    """
    if not row['error'] and row['condition'] == 'unseen':
        cond = 'correct_rej'
    elif not row['error'] and row['condition'] == 'seen':
        cond = 'hit'
    elif row['error'] is True and row['condition'] == 'unseen':
        cond = 'false_alarm'
    elif row['error'] is True and row['condition'] == 'seen':
        cond = 'miss'
    else:
        cond = pd.NA
    return cond


def subf(row):
    """
    # TODO : Double-check that these are being correctly generated
    """
    view_cond, pres_cond = row['subcondition'].split('-', 1)
    if '-' in pres_cond:
        pres_cond = pres_cond.split('-')[-1]
    if not row['error'] and view_cond == 'unseen':
        if pres_cond == 'within':
            cond = 'correct_rej-within'
        elif pres_cond == 'between':
            cond = 'correct_rej-between'
    elif not row['error'] and view_cond == 'seen':
        if pres_cond == 'within':
            cond = 'hit-within'
        elif pres_cond == 'between':
            cond = 'hit-between'
    elif row['error'] is True and view_cond == 'unseen':
        if pres_cond == 'within':
            cond = 'false_alarm-within'
        elif pres_cond == 'between':
            cond = 'false_alarm-between'
    elif row['error'] is True and view_cond == 'seen':
        if pres_cond == 'within':
            cond = 'miss-within'
        elif pres_cond == 'between':
            cond = 'miss-between'
    else:
        cond = pd.NA
    return cond


# only grab sessions with three digits ; indicates corrected information
events = sorted(Path(
    'things.fmriprep', 'sourcedata', 'things', 'sub-01'
).rglob('*ses-???_*events.tsv'))
# drop ses-001 from all analyses
events = list(filter( lambda e: ('ses-001' not in str(e)), events))

sessions, runs = [], []
hit_within, hit_between = [], []
rej_within, rej_between = [], []
for event in events:

    # load in events files and create memory conditions
    # based on performance
    df = pd.read_csv(event, sep='\t')
    df = df[~df.exclude_session]  # drop 'exclude_session' 
    # keep only high-confidence responses
    df['response_confidence'] = df['response_confidence'].fillna(False)
    df = df[df.response_confidence]
    df['memory_cond'] = df.apply(subf, axis=1)
    memory_counts = df['memory_cond'].value_counts(sort=False)

    _, ses, _, run, _ = event.name.split('_')
    sessions.append(ses)
    runs.append(run)

    cond_keys = ('hit-within', 'correct_rej-within', 'hit-between', 'correct_rej-between')
    cond_arrays = (hit_within, rej_within, hit_between, rej_between)

    for cond_array, cond_key in zip(cond_arrays, cond_keys):
        try:
            cond_array.append(memory_counts[cond_key])
        except KeyError:
            cond_array.append(0)
            pass

counts_df = pd.DataFrame({
    'sessions': sessions,
    'runs': runs,
    'hits-within': hit_within,
    'hits-between': hit_between,
    'rejections': [sum(x) for x in zip(rej_within, rej_between)],
})

# plt.figure()
# sns.jointplot(counts_df, x='hits', y='rejections')

plt.figure()
ax = sns.boxplot(counts_df['hits-within'], color='forestgreen', boxprops={'alpha': 0.75}, orient='h')
sns.stripplot(counts_df['hits-within'], color='forestgreen', dodge=True, orient='h', ax=ax)
plt.title('Counts: High Confidence')

# plt.figure()
# ax = sns.boxplot(counts_df['hits-between'], color='forestgreen', boxprops={'alpha': 0.75}, orient='h')
# sns.stripplot(counts_df['hits-between'], color='forestgreen', dodge=True, orient='h', ax=ax)
# plt.title('Counts: High Confidence')


# plt.figure()
# ax = sns.boxplot(counts_df['rejections'], color='firebrick', boxprops={'alpha': 0.75}, orient='h')
# sns.stripplot(counts_df['rejections'], color='firebrick', dodge=True, orient='h', ax=ax)
# plt.title('Counts: All Confidence')
plt.show()
