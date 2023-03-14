# Importing libraries.

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, precision_score, ConfusionMatrixDisplay, classification_report
from yellowbrick.target import class_balance
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import time
from imblearn.ensemble import RUSBoostClassifier, EasyEnsembleClassifier, \
    BalancedBaggingClassifier, BalancedRandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings("ignore")


# Uploading the dataset.


def load_dataset(filename, index_col=None):
    dataset = pd.read_csv(f'{filename}', index_col=index_col)
    pd.options.display.max_columns = None
    return dataset


url = 'https://raw.githubusercontent.com/jakubtwalczak/Players_classification_20_21/main/Top5leagues.csv'

df = load_dataset(filename=url)
print(df.head())
print(df.tail())

# First descriptive statistics.

print(df.info())
print(df.describe().T)
print(df.columns)

# Conversion of data frame columns with height and weight into numeric.

df['Height'] = df['Height'].str.replace("\D", "", regex=True).astype(float)
df['Weight'] = df['Weight'].str.replace("\D", "", regex=True).astype(float)

# Checking which columns contain missing values.

print(df.isnull().any())

# Replacing NaN-s from the height and weight columns with a robust median values.

df['Weight'].fillna((df['Weight'].median()), inplace=True)
df['Height'].fillna((df['Height'].median()), inplace=True)

# Replacing the rest of missing values with zeros.

df.fillna(0, inplace=True)

# Deletion of unneeded columns.

df.drop(columns=['Name', 'Duels total'], axis=1, inplace=True)

# Limitation the players' number - we're taking into consideration
# only the players with number of minutes equivalent to 5 full matches.

df = df[df['Total minutes'] >= 450]

# Last look at the descriptive statistics after the "cleaning" of data.

print(df.info())
print(df.describe().T)

# Displaying the classes balance.

print(df['Position'].value_counts(dropna=False, normalize=True))
sns.countplot(x='Position', data=df, palette='GnBu_d')
plt.show()

# Descriptive statistics for each class

pivot = pd.pivot_table(df, index='Position', values=['Height', 'Weight', 'Apps', 'Total minutes',
                                                     'Scored goals', 'Assists', 'Total shots', 'Shots on goal',
                                                     'Duels won', 'Dribble attempts', 'Dribbles succ.',
                                                     'Total duels', 'Total passes', 'Key passes', 'Tackles', 'Blocks',
                                                     'Interceptions', 'Total saves', 'Goals conceded', 'Fouls drawn',
                                                     'Fouls committed', 'Yellow cards', 'Red cards for 2nd yellow',
                                                     'Straight red cards', 'Penalties won', 'Penalties scored',
                                                     'Penalties missed', 'Penalties committed', 'Penalties saved'],
                       aggfunc=[np.mean, np.median, min, max, np.std])
pd.options.display.max_columns = None
print(pivot)

# Correlation heatmap.

cm = df.corr()
fig, ax = plt.subplots(figsize=(30, 20))
sns.heatmap(cm, annot=True)

# BINARY CLASSIFICATION.

# Sets division into features and classes.

X = df.drop(columns='Position')
Y = df['Position']
print(X.head())
print(Y)

# Classes encoding, displaying balance.

y = np.where(Y == 'Attacker', 1, 0)
print(y)
print(np.unique(y, return_counts=True))
class_balance(y.flatten(), labels=['Non-forwards', 'Forwards'])

# Train test split

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.3,
                                                    random_state=42,
                                                    stratify=y)

print(np.unique(y_train, return_counts=True))
print(np.unique(y_test, return_counts=True))

# PCA.

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

pca = PCA()

X_train_pca = pca.fit_transform(X_train_sc)
X_test_pca = pca.transform(X_test_sc)

# Variance explanation plot.

features = range(1, pca.n_components_ + 1)
plt.bar(features, pca.explained_variance_)
plt.xlim(xmin=0.5)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.show()

# Displaying the variance values of features.

print(pca.explained_variance_ratio_)

# PCA fitting with 14 features.

pca = PCA(n_components=14)

X_train_sc = pca.fit_transform(X_train_sc)
X_test_sc = pca.transform(X_test_sc)

print(X_train_sc.shape)

# Next variance explanation plot.

features = range(1, pca.n_components_ + 1)
plt.bar(features, pca.explained_variance_)
plt.xlim(xmin=0.5)
plt.xlabel('PCA feature')
plt.ylabel('variance')
plt.show()
print(np.sum(pca.explained_variance_ratio_))

# MODELLING.
# Convention of modelling pipeline (refers to both classification tasks):
# 1. Set the instance of classifier
# 2. If a model doesn't use undersampling - oversample the minority class with SMOTE.
# 3. Train it with default parameters
# 4. Display the model's metrics
# 5. Draw the confusion matrix as a plot
# 6. Create and append lists to compare models' recall, precision and training time.

# Balanced Random Forest Classifier.

brfc_model = BalancedRandomForestClassifier()

start = time.time()
brfc_model.fit(X_train_sc, y_train)
brfc_train_pred = brfc_model.predict(X_train_sc)
stop = time.time()
time_diff_brfc = stop - start
print(f"Training time: {time_diff_brfc} s.")

brfc_test_pred = brfc_model.predict(X_test_sc)
print(f'Training metrics:\n\n{classification_report(y_train, brfc_train_pred)}')
print('_____________________________________________________\n')
print(f'Test metrics:\n\n{classification_report(y_test, brfc_test_pred)}')

labels = ['Other positions', 'Forwards']


def conf_matrix_show(estimator, feats, lbls, classes):
    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_estimator(estimator,
                                          feats,
                                          lbls,
                                          display_labels=classes,
                                          ax=ax,
                                          values_format=' ')
    ax.grid(False)


conf_matrix_show(brfc_model, X_train_sc, y_train, labels)
conf_matrix_show(brfc_model, X_test_sc, y_test, labels)

train_recall = []
test_recall = []
train_precision = []
test_precision = []
time_diffs = []
train_recall.append(recall_score(y_train, brfc_train_pred))
test_recall.append(recall_score(y_test, brfc_test_pred))
train_precision.append(precision_score(y_train, brfc_train_pred))
test_precision.append(precision_score(y_test, brfc_test_pred))
time_diffs.append(time_diff_brfc)

# EasyEnsembleClassifier.

eec_model = EasyEnsembleClassifier()

start = time.time()
eec_model.fit(X_train_sc, y_train)
eec_train_pred = eec_model.predict(X_train_sc)
stop = time.time()
time_diff_eec = stop - start
print(f"Training time: {time_diff_eec} s")

eec_test_pred = eec_model.predict(X_test_sc)
print(f'Training metrics:\n\n{classification_report(y_train, eec_train_pred)}')
print('_____________________________________________________\n')
print(f'Test metrics:\n\n{classification_report(y_test, eec_test_pred)}')

conf_matrix_show(eec_model, X_train_sc, y_train, labels)
conf_matrix_show(eec_model, X_test_sc, y_test, labels)

train_recall.append(recall_score(y_train, eec_train_pred))
test_recall.append(recall_score(y_test, eec_test_pred))
train_precision.append(precision_score(y_train, eec_train_pred))
test_precision.append(precision_score(y_test, eec_test_pred))
time_diffs.append(time_diff_eec)

# RUS Boost Classifier.

rusbc_model = RUSBoostClassifier()

start = time.time()
rusbc_model.fit(X_train_sc, y_train)
rusbc_train_pred = rusbc_model.predict(X_train_sc)
stop = time.time()
time_diff_rusbc = stop - start
print(f"Training time: {time_diff_rusbc} s")

rusbc_test_pred = rusbc_model.predict(X_test_sc)
print(f'Training metrics:\n\n{classification_report(y_train, rusbc_train_pred)}')
print('_____________________________________________________\n')
print(f'Test metrics:\n\n{classification_report(y_test, rusbc_test_pred)}')

conf_matrix_show(rusbc_model, X_train_sc, y_train, labels)
conf_matrix_show(rusbc_model, X_test_sc, y_test, labels)

train_recall.append(recall_score(y_train, rusbc_train_pred))
test_recall.append(recall_score(y_test, rusbc_test_pred))
train_precision.append(precision_score(y_train, rusbc_train_pred))
test_precision.append(precision_score(y_test, rusbc_test_pred))
time_diffs.append(time_diff_rusbc)

# Balanced Bagging Classifier.

bbc_model = BalancedBaggingClassifier()

start = time.time()
bbc_model.fit(X_train_sc, y_train)
bbc_train_pred = bbc_model.predict(X_train_sc)
stop = time.time()
time_diff_bbc = stop - start
print(f"Training time: {time_diff_bbc} s")

bbc_test_pred = bbc_model.predict(X_test_sc)
print(f'Training metrics:\n\n{classification_report(y_train, bbc_train_pred)}')
print('_____________________________________________________\n')
print(f'Test metrics:\n\n{classification_report(y_test, bbc_test_pred)}')

conf_matrix_show(bbc_model, X_train_sc, y_train, labels)
conf_matrix_show(bbc_model, X_test_sc, y_test, labels)

train_recall.append(recall_score(y_train, bbc_train_pred))
test_recall.append(recall_score(y_test, bbc_test_pred))
train_precision.append(precision_score(y_train, bbc_train_pred))
test_precision.append(precision_score(y_test, bbc_test_pred))
time_diffs.append(time_diff_bbc)

# AdaBoost Classifier + SMOTE.

oversampler = SMOTE()
X_over_sc, y_over = oversampler.fit_resample(X_train_sc, y_train)

print(f'Oversampled X_train shape: {X_over_sc.shape}.')
print(f'Oversampled y_train shape: {y_over.shape}.')
class_balance(y_over.flatten(), labels=['Non-forwards', 'Forwards'])

abc_model = AdaBoostClassifier()

start = time.time()
abc_model.fit(X_over_sc, y_over)
abc_train_pred = abc_model.predict(X_over_sc)
stop = time.time()
time_diff_abc = stop - start
print(f"Training time: {time_diff_abc} s.")

abc_test_pred = abc_model.predict(X_test_sc)
print(f'Training metrics:\n\n{classification_report(y_over, abc_train_pred)}')
print('_____________________________________________________\n')
print(f'Test metrics:\n\n{classification_report(y_test, abc_test_pred)}')

conf_matrix_show(abc_model, X_over_sc, y_over, labels)
conf_matrix_show(abc_model, X_test_sc, y_test, labels)

train_recall.append(recall_score(y_over, abc_train_pred))
test_recall.append(recall_score(y_test, abc_test_pred))
train_precision.append(precision_score(y_over, abc_train_pred))
test_precision.append(precision_score(y_test, abc_test_pred))
time_diffs.append(time_diff_abc)

# Random Forest Classifier + SMOTE.

rfc_model = RandomForestClassifier()

start = time.time()
rfc_model.fit(X_over_sc, y_over)
rfc_train_pred = rfc_model.predict(X_over_sc)
stop = time.time()
time_diff_rfc = stop - start
print(f"Training time: {time_diff_rfc} s.")

rfc_test_pred = rfc_model.predict(X_test_sc)
print(f'Training metrics:\n\n{classification_report(y_over, rfc_train_pred)}')
print('_____________________________________________________\n')
print(f'Test metrics:\n\n{classification_report(y_test, rfc_test_pred)}')

conf_matrix_show(rfc_model, X_over_sc, y_over, labels)
conf_matrix_show(rfc_model, X_test_sc, y_test, labels)

train_recall.append(recall_score(y_over, rfc_train_pred))
test_recall.append(recall_score(y_test, rfc_test_pred))
train_precision.append(precision_score(y_over, rfc_train_pred))
test_precision.append(precision_score(y_test, rfc_test_pred))
time_diffs.append(time_diff_rfc)

# Summary - creation of summarizing data frame.

scores = {
    'Training recall': [i for i in train_recall],
    'Test recall': [i for i in test_recall],
    'Training precision': [i for i in train_precision],
    'Test precision': [i for i in test_precision],
    'Training time': [i for i in time_diffs]
}

scores = pd.DataFrame(scores, index=['Balanced Random Forest', 'Easy Ensemble',
                                     'RUS Boost', 'Balanced Bagging', 'Ada Boost + SMOTE', 'Random Forest + SMOTE'])
scores['Recall difference'] = scores['Training recall'] - scores['Test recall']
scores['Precision difference'] = scores['Training precision'] - scores['Test precision']
(scores.style.highlight_max(color='lightgreen', axis=0)
 .highlight_min(color='lightblue', axis=0))

# MULTICLASS CLASSIFICATION.

# Classes encoding.

le = LabelEncoder()
y_l = le.fit_transform(Y)
print(le.classes_)
print(np.unique(y_l))

# Train test split.

X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y_l,
                                                    test_size=0.3,
                                                    random_state=42, stratify=y_l)

# PCA, normalization of data, SMOTE.

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

X_train_sc = pca.fit_transform(X_train_sc)
X_test_sc = pca.transform(X_test_sc)

X_res_sc, y_res = oversampler.fit_resample(X_train_sc, y_train)
print(f'Oversampled X_train shape: {X_res_sc.shape}.')
print(f'Oversampled y_train shape: {y_res.shape}.')

class_balance(y_res.flatten(), labels=['Attackers', 'Defenders', 'Goalkeepers', 'Midfielders'])

# MODELLING (pipeline the same as the first time).
# Balanced Random Forest Classifier.

brfc_model = BalancedRandomForestClassifier()

start = time.time()
brfc_model.fit(X_train_sc, y_train)
brfc_train_pred = brfc_model.predict(X_train_sc)
stop = time.time()
time_diff_brfc = stop - start
print(f"Training time: {time_diff_brfc} s.")

brfc_test_pred = brfc_model.predict(X_test_sc)
print(f'Training metrics:\n\n{classification_report(y_train, brfc_train_pred)}')
print('_____________________________________________________\n')
print(f'Test metrics:\n\n{classification_report(y_test, brfc_test_pred)}')

labels = ['Attackers', 'Defenders', 'Goalkeepers', 'Midfielders']

conf_matrix_show(brfc_model, X_train_sc, y_train, labels)
conf_matrix_show(brfc_model, X_test_sc, y_test, labels)

train_recall = []
test_recall = []
train_precision = []
test_precision = []
time_diffs = []
train_recall.append(recall_score(y_train, brfc_train_pred, average='weighted'))
test_recall.append(recall_score(y_test, brfc_test_pred, average='weighted'))
train_precision.append(precision_score(y_train, brfc_train_pred, average='weighted'))
test_precision.append(precision_score(y_test, brfc_test_pred, average='weighted'))
time_diffs.append(time_diff_brfc)

# Easy Ensemble Classifier.

eec_model = EasyEnsembleClassifier()

start = time.time()
eec_model.fit(X_train_sc, y_train)
eec_train_pred = eec_model.predict(X_train_sc)
stop = time.time()
time_diff_eec = stop - start
print(f"Training time: {time_diff_eec} s")

eec_test_pred = eec_model.predict(X_test_sc)
print(f'Training metrics:\n\n{classification_report(y_train, eec_train_pred)}')
print('_____________________________________________________\n')
print(f'Test metrics:\n\n{classification_report(y_test, eec_test_pred)}')

conf_matrix_show(eec_model, X_train_sc, y_train, labels)
conf_matrix_show(eec_model, X_test_sc, y_test, labels)

train_recall.append(recall_score(y_train, eec_train_pred, average='weighted'))
test_recall.append(recall_score(y_test, eec_test_pred, average='weighted'))
train_precision.append(precision_score(y_train, eec_train_pred, average='weighted'))
test_precision.append(precision_score(y_test, eec_test_pred, average='weighted'))
time_diffs.append(time_diff_eec)

# RUS Boost Classifier.

rusbc_model = RUSBoostClassifier()

start = time.time()
rusbc_model.fit(X_train_sc, y_train)
rusbc_train_pred = rusbc_model.predict(X_train_sc)
stop = time.time()
time_diff_rusbc = stop - start
print(f"Training time: {time_diff_rusbc} s")

rusbc_test_pred = rusbc_model.predict(X_test_sc)
print(f'Training metrics:\n\n{classification_report(y_train, rusbc_train_pred)}')
print('_____________________________________________________\n')
print(f'Test metrics:\n\n{classification_report(y_test, rusbc_test_pred)}')

conf_matrix_show(rusbc_model, X_train_sc, y_train, labels)
conf_matrix_show(rusbc_model, X_test_sc, y_test, labels)

train_recall.append(recall_score(y_train, rusbc_train_pred, average='weighted'))
test_recall.append(recall_score(y_test, rusbc_test_pred, average='weighted'))
train_precision.append(precision_score(y_train, rusbc_train_pred, average='weighted'))
test_precision.append(precision_score(y_test, rusbc_test_pred, average='weighted'))
time_diffs.append(time_diff_rusbc)

# Balanced Bagging Classifier.

bbc_model = BalancedBaggingClassifier()

start = time.time()
bbc_model.fit(X_train_sc, y_train)
bbc_train_pred = bbc_model.predict(X_train_sc)
stop = time.time()
time_diff_bbc = stop - start
print(f"Training time: {time_diff_bbc} s")

bbc_test_pred = bbc_model.predict(X_test_sc)
print(f'Training metrics:\n\n{classification_report(y_train, bbc_train_pred)}')
print('_____________________________________________________\n')
print(f'Test metrics:\n\n{classification_report(y_test, bbc_test_pred)}')

conf_matrix_show(bbc_model, X_train_sc, y_train, labels)
conf_matrix_show(bbc_model, X_test_sc, y_test, labels)

train_recall.append(recall_score(y_train, bbc_train_pred, average='weighted'))
test_recall.append(recall_score(y_test, bbc_test_pred, average='weighted'))
train_precision.append(precision_score(y_train, bbc_train_pred, average='weighted'))
test_precision.append(precision_score(y_test, bbc_test_pred, average='weighted'))
time_diffs.append(time_diff_bbc)

# AdaBoost Classifier + SMOTE.

abc_model = AdaBoostClassifier()

start = time.time()
abc_model.fit(X_res_sc, y_res)
abc_train_pred = abc_model.predict(X_res_sc)
stop = time.time()
time_diff_abc = stop - start
print(f"Training time: {time_diff_abc} s.")

abc_test_pred = abc_model.predict(X_test_sc)
print(f'Training metrics:\n\n{classification_report(y_res, abc_train_pred)}')
print('_____________________________________________________\n')
print(f'Test metrics:\n\n{classification_report(y_test, abc_test_pred)}')

conf_matrix_show(abc_model, X_res_sc, y_res, labels)
conf_matrix_show(abc_model, X_test_sc, y_test, labels)

train_recall.append(recall_score(y_res, abc_train_pred, average='weighted'))
test_recall.append(recall_score(y_test, abc_test_pred, average='weighted'))
train_precision.append(precision_score(y_res, abc_train_pred, average='weighted'))
test_precision.append(precision_score(y_test, abc_test_pred, average='weighted'))
time_diffs.append(time_diff_abc)

# Random Forest Classifier + SMOTE.

rfc_model = RandomForestClassifier()

start = time.time()
rfc_model.fit(X_res_sc, y_res)
rfc_train_pred = rfc_model.predict(X_res_sc)
stop = time.time()
time_diff_rfc = stop - start
print(f"Training time: {time_diff_rfc} s.")

rfc_test_pred = rfc_model.predict(X_test_sc)
print(f'Training metrics:\n\n{classification_report(y_res, rfc_train_pred)}')
print('_____________________________________________________\n')
print(f'Test metrics:\n\n{classification_report(y_test, rfc_test_pred)}')

conf_matrix_show(rfc_model, X_res_sc, y_res, labels)
conf_matrix_show(rfc_model, X_test_sc, y_test, labels)

train_recall.append(recall_score(y_res, rfc_train_pred, average='weighted'))
test_recall.append(recall_score(y_test, rfc_test_pred, average='weighted'))
train_precision.append(precision_score(y_res, rfc_train_pred, average='weighted'))
test_precision.append(precision_score(y_test, rfc_test_pred, average='weighted'))
time_diffs.append(time_diff_rfc)

# Summary.

scores = {
    'Training recall': [i for i in train_recall],
    'Test recall': [i for i in test_recall],
    'Training precision': [i for i in train_precision],
    'Test precision': [i for i in test_precision],
    'Training time': [i for i in time_diffs]
}

scores = pd.DataFrame(scores, index=['Balanced Random Forest', 'Easy Ensemble',
                                     'RUS Boost', 'Balanced Bagging', 'Ada Boost + SMOTE', 'Random Forest + SMOTE'])
scores['Recall difference'] = scores['Training recall'] - scores['Test recall']
scores['Precision difference'] = scores['Training precision'] - scores['Test precision']
(scores.style.highlight_max(color='lightgreen', axis=0)
 .highlight_min(color='lightblue', axis=0))
