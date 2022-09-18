import warnings
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.svm import SVC
from matplotlib import pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, BatchNormalization, Dropout
sns.set_theme()
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# neural network class
# =====================================================================================================================

class NeuralNetwork:
    """Neural Network wrapper around Keras, making it consistent with sklearn models' fit and predict methods."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.network = Sequential(
            [
                Dense(200, activation='relu'),  # first hidden layer
                Dropout(rate=0.5),
                BatchNormalization(),
                Dense(100, activation='relu'),  # second hidden layer
                Dropout(rate=0.5),
                BatchNormalization(),
                Dense(1, activation='sigmoid')  # output layer
            ]
        )
        self.network.compile(optimizer='adam', loss='binary_crossentropy')

    def fit(self, X, y):
        X = self.scaler.fit_transform(X)  # feature standardization
        self.network.fit(x=X, y=y, batch_size=32, epochs=40)
        return self

    def predict(self, X, p=0.5):
        X = self.scaler.transform(X)  # feature standardization
        y_prob = self.network.predict(X)  # "score" or predicted probability of positive label: P(y=1 | x)
        return (y_prob > p).astype(int).squeeze()  # label prediction


# main
# =====================================================================================================================

# loading data set
statistics = ('mean', 'std', 'kurt', 'skew')
curves = ('ip', 'dmsnr')
col_names = [f'{stat}_{curve}' for curve in curves for stat in statistics]
col_names.append('label')
df = pd.read_csv('HTRU_2.csv', names=col_names)
feature_names = {
    0: 'Mean of the integrated profile',
    1: 'Standard deviation of the integrated profile',
    2: 'Excess kurtosis of the integrated profile',
    3: 'Skewness of the integrated profile',
    4: 'Mean of the DM-SNR curve',
    5: 'Standard deviation of the DM-SNR curve',
    6: 'Excess kurtosis of the DM-SNR curve',
    7: 'Skewness of the DM-SNR curve',
}
print('\nCount of missing values per column:')
print(df.isna().sum())  # no missing values in data set

# data set split
test_size = val_size = round(0.2 * df.shape[0])  # 60-20-20 train-val-test split
np.random.seed(0)  # setting random seed for fixed train-val-test sets
df_temp, df_test = train_test_split(df, test_size=test_size, stratify=df['label'])  # test
df_train, df_val = train_test_split(df_temp, test_size=val_size, stratify=df_temp['label'])  # train-val
np.random.seed()  # resetting random seed

# plot variables
selected_columns = (0, 2, 3)  # features more highly correlated with label (shown later in correlation heatmap)
colors = ('r', 'b')  # red for negative labels, blue for positive labels. green used when not stratifying.
label_names = ('RFI / noise', 'Pulsars')

# Q-Q plots
quantiles = np.arange(0.05, 1, 0.05)
for col in selected_columns:
    x1 = np.quantile(df_train.iloc[:, col], quantiles)
    x2 = np.quantile(df_val.iloc[:, col], quantiles)
    sns.scatterplot(x=x1, y=x2, color='g')
    plt.gca().axline((x1[0], x1[0]), slope=1, color='r', alpha=0.7)  # 45° line
    plt.title(f'Q-Q Plot: {feature_names[col]}')
    plt.xlabel('Training set')
    plt.ylabel('Validation set')
    plt.tight_layout()
    # plt.savefig(f'images/QQ_{col_names[col]}')
    plt.show()
    # plt.close()

# histograms
limits = [(None, None), (-1, 2), (-2, 4)]  # excluding outliers from visualization (for better scale)
for col, (left, right) in zip(selected_columns, limits):
    feature = col_names[col]
    sns.histplot(data=df_train, x=feature, stat='percent', color='g')
    plt.title('Histogram')
    plt.xlabel(feature_names[col])
    plt.xlim(left=left, right=right)
    plt.tight_layout()
    # plt.savefig(f'images/hist_{feature}')
    plt.show()
    # plt.close()

# histograms stratified by label
for col, bin_width in zip(selected_columns, (10, 0.2, 1)):
    feature = col_names[col]
    for label in (0, 1):
        sns.histplot(data=df_train[df_train['label'] == label], x=feature, color=colors[label], stat='percent',
                     label=label_names[label], binwidth=bin_width)
    plt.title('Stratified Histogram')
    plt.xlabel(feature_names[col])
    plt.legend(facecolor='white')
    plt.tight_layout()
    # plt.savefig(f'images/hist_strat_{feature}')
    plt.show()
    # plt.close()

# dependence visualization: correlation heatmap
sns.heatmap(df_train.corr(), annot=True, fmt='.2f', cmap='vlag')
plt.title('Correlation Heatmap')
plt.tight_layout()
# plt.savefig('images/corr_heatmap')
plt.show()
# plt.close()

# data description
print('\nDescriptive statistics:\n')
pd.options.display.float_format = '{:,.2f}'.format
print(df_train.describe().to_string())

# box and violin plots
for col in selected_columns:
    feature = col_names[col]
    for plot_type, plot_name in ((sns.boxplot, 'box'), (sns.violinplot, 'violin')):
        plot_type(data=df_train, x='label', y=feature, palette=colors)
        plt.xticks((0, 1), label_names)
        plt.gca().set(xlabel=None)
        plt.title(feature_names[col])
        # plt.savefig(f'images/{plot_name}_{feature}')
        plt.show()
        # plt.close()

# data exploration visualization: pairwise scatter plots (grouped by curve: IP or DM-SNR)
for curve in curves:
    features = [f'{stat}_{curve}' for stat in statistics]
    features.append('label')
    sns.pairplot(data=df_train[features], hue='label', palette=colors, plot_kws={'s': 10})
    # plt.savefig(f'images/pairwise_{curve}')
    plt.show()
    # plt.close()

# hypothesis testing - whether 2 samples come from the same distribution
print('\nHypothesis testing (alpha = 0.05)')
print('Null hypothesis: variables from negative and positive examples follow the same distribution')
for col in selected_columns:
    print(f'\n\t{feature_names[col]}')
    feature = col_names[col]
    x = df_train.loc[df_train['label'] == 0, feature]  # negative examples (RFI noise)
    y = df_train.loc[df_train['label'] == 1, feature]  # positive examples (Pulsars)
    cvm = stats.cramervonmises_2samp(x, y).pvalue
    ks = stats.ks_2samp(x, y).pvalue
    ad = stats.anderson_ksamp([x, y]).significance_level
    print(f'\t\tCramér-von Mises p-value: {cvm:.2f}.')
    print(f'\t\tKolmogorov-Smirnov p-value: {ks:.2f}')
    print(f'\t\tAnderson-Darling approximate significance level: {ad:.2f}')

# probability distributions (stratified)
print('\nProbability distributions')
dist_names = ['norm', 'beta', 'gamma']  # candidate distributions
for col in selected_columns:
    feature = col_names[col]
    plt.figure(figsize=(12, 6))
    print(f'\n\t{feature_names[col]}:')
    for label in (0, 1):
        plt.subplot(1, 2, label + 1)
        plt.title(label_names[label])
        x = df_train.loc[df_train['label'] == label, feature]
        g = sns.histplot(x=x, stat='density', color=colors[label], alpha=0.6)
        max_y = 1.2 * max([rectangle.get_height() for rectangle in g.patches])
        plt.ylim(top=max_y)
        q = np.linspace(x.min(), x.max(), 100)
        for dist_name in dist_names:
            dist = getattr(stats, dist_name)
            params = dist.fit(x)  # fitted distribution parameters
            sns.lineplot(x=q, y=dist.pdf(q, *params), label=dist_name)  # fitted distribution pdf
            if dist_name == 'beta':  # beta distribution seems to be the best fit in all cases
                print(f'\t\t{label_names[label]}: Beta distribution (alpha = {params[0]:.2e}, beta = {params[1]:.2e}, '
                      f'loc = {params[2]:.2e}, scale = {params[3]:.2e})')
        plt.legend(facecolor='white')
        plt.xlabel(feature_names[col])
    plt.suptitle(feature_names[col])
    plt.tight_layout()
    # plt.savefig(f'images/dist_{col_names[col]}')
    plt.show()
    # plt.close()

# hypothesis testing: two-sample t-test
print('\nHypothesis testing (alpha = 0.05)')  # two-sided
print('Null hypothesis: variables from negative and positive examples have the same mean')
for col in selected_columns:
    print(f'\n\t{feature_names[col]}')
    feature = col_names[col]
    x = df_train.loc[df_train['label'] == 0, feature]  # sample 1: negative examples (RFI noise)
    y = df_train.loc[df_train['label'] == 1, feature]  # sample 2: positive examples (Pulsars)
    mu_x, sigma_x, N_x = x.mean(), x.std(), x.shape[0]
    mu_y, sigma_y, N_y = y.mean(), y.std(), y.shape[0]
    temp1 = (sigma_x ** 2) / N_x
    temp2 = (sigma_y ** 2) / N_y
    t0 = (mu_x - mu_y) / ((temp1 + temp2) ** 0.5)  # test statistic
    term1 = (temp1 + temp2) ** 2
    term2 = (temp1 ** 2) / (N_x - 1)
    term3 = (temp2 ** 2) / (N_y - 1)
    ddof = term1 / (term2 + term3)  # degrees of freedom
    p = 2 * min(stats.t.cdf(t0, df=ddof), 1 - stats.t.cdf(t0, df=ddof))  # p-value
    print(f'\t\t2-sample t-test p-value: {p:.2f}')
# note: we are aware of the stats.ttest_ind() function which could have simplified this section. we merely wanted to
# perform the calculations manually. during development we tested that the results matched those of the scipy library.

# preparing data for training
X_train, y_train = df_train.iloc[:, :-1].to_numpy(), df_train.iloc[:, -1].to_numpy()
X_val, y_val = df_val.iloc[:, :-1].to_numpy(), df_val.iloc[:, -1].to_numpy()
X_test, y_test = df_test.iloc[:, :-1].to_numpy(), df_test.iloc[:, -1].to_numpy()

# model training
model_dict = {
    'Gaussian naive Bayes': GaussianNB(),
    'K-nearest neighbors': KNeighborsClassifier(),
    'Support vector machine': SVC(),
    'Decision Tree': DecisionTreeClassifier(criterion='entropy'),
    'Random Forest': RandomForestClassifier(),
    'Neural Network': NeuralNetwork(),
}
for model in model_dict.values():
    model.fit(X_train, y_train)

# models performance on validation set
print('\nResults on validation set:\n')
results = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1_score'])
for model_name, model in model_dict.items():
    y_pred = model.predict(X_val)
    results = results.append(
        {
            'model': model_name,
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1_score': f1_score(y_val, y_pred),
        },
        ignore_index=True
    )
print(results.to_string(index=False))

# comparing two classifiers
print('\nClassifier comparison (McNemar test, alpha = 0.05):')
print('Null hypothesis: the two classifiers perform equally well on the validation set')
model_list = list(model_dict.keys())
for i, name1 in enumerate(model_list):
    model1 = model_dict[name1]
    for name2 in model_list[i + 1:]:
        model2 = model_dict[name2]
        y_pred1 = model1.predict(X_val)
        y_pred2 = model2.predict(X_val)
        correct1 = y_pred1 == y_val
        correct2 = y_pred2 == y_val
        n01 = np.logical_and((correct1 == 0), (correct2 == 1)).sum()
        n10 = np.logical_and((correct1 == 1), (correct2 == 0)).sum()
        if n01 + n10 <= 25:  # mcnemar exact test
            s0 = min(n01, n10)
            p_value = 2 * stats.binom.cdf(s0, n01 + n10, 0.5)
        else:  # mcnemar approximate test
            s0 = ((abs(n01 - n10) - 1) ** 2) / (n01 + n10)
            p_value = 1 - stats.chi2.cdf(s0, df=1)
        print(f'\t{name1} vs {name2}: {p_value = :.2f}')

# selected model (KNN) performance on test set
print(f'\nResults on test set:\n')
model_name = 'K-nearest neighbors'
model = model_dict[model_name]
y_pred = model.predict(X_test)
results = pd.DataFrame(columns=['model', 'accuracy', 'precision', 'recall', 'f1_score'])
results = results.append(
    {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1_score': f1_score(y_test, y_pred),
    },
    ignore_index=True
)
print(results.to_string(index=False))


# impact of neural network's prediction threshold on metrics
net = model_dict['Neural Network']  # trained neural network
p_arr = np.arange(.05, 1, .05)  # threshold values (between 0 and 1)
f1_arr, prec_arr, recall_arr = np.zeros_like(p_arr), np.zeros_like(p_arr), np.zeros_like(p_arr)
for i, p in enumerate(p_arr):
    y_pred = net.predict(X_val, p)
    f1_arr[i] = f1_score(y_val, y_pred)
    prec_arr[i] = precision_score(y_val, y_pred)
    recall_arr[i] = recall_score(y_val, y_pred)
f_max = np.max(f1_arr)  # max f1-score
p_max = p_arr[np.argmax(f1_arr)]  # threshold value that maximizes f1-score
# plotting metrics vs threshold values
sns.lineplot(x=p_arr, y=f1_arr, label='F1-score')
sns.lineplot(x=p_arr, y=prec_arr, label='Precision')
sns.lineplot(x=p_arr, y=recall_arr, label='Recall')
plt.hlines(y=f_max, xmin=0, xmax=p_max, ls='--')  # maximum f1-score
plt.vlines(x=p_max, ymin=0, ymax=f_max, ls='--')  # threshold value that maximizes f1-score
plt.legend(facecolor='white', loc='lower right')
plt.title('Metrics vs Threshold')
plt.xlabel('Threshold')
plt.ylabel('Metric value')
plt.xlim(left=0)
plt.ylim(bottom=0)
# plt.savefig('images/threshold_impact')
plt.show()
# plt.close()
