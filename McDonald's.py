import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from statsmodels.graphics.mosaicplot import mosaic
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from joblib import Parallel, delayed
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
np.set_printoptions(threshold=np.inf)
pd.set_option('expand_frame_repr', False)


#                                    STEP 4: EXPLORING DATA


mcdonalds = pd.read_csv('mcdonalds.csv')
print("Attribues of mcdonald's dataset: ")
print(mcdonalds.columns)
print("Dimention of dataset: ", mcdonalds.shape)
print("First ten rows of dataset: ")
print(mcdonalds[:10])

md_x = mcdonalds.iloc[:, :11].values
print(md_x[:10])
md_x = (md_x == "Yes").astype(int)
print(md_x[:10])
col_mean = np.round(np.mean(md_x, axis=0), 2)
col_mean1 = pd.DataFrame(col_mean, mcdonalds.columns[:11])
print("Mean of all values for particular attribute: ", col_mean1[:10])

pca = PCA()
md_pca = pca.fit(md_x)
std_dev = np.sqrt(md_pca.explained_variance_)
proportion_variance = md_pca.explained_variance_ratio_
cumulative_variance = np.cumsum(proportion_variance)

pca_summary = pd.DataFrame({
    "principal component": [f'PC{i+1}' for i in range(len(proportion_variance))],
    "standard deviation": np.round(std_dev, 2),
    "proportion variance": np.round(proportion_variance, 2),
    "cumulative variance": np.round(cumulative_variance, 2)
})
print(pca_summary)

rotation_matrix = md_pca.components_.T
rotation_df = pd.DataFrame(
    rotation_matrix,
    columns=[f'PC{i+1}' for i in range(rotation_matrix.shape[1])],
    index=mcdonalds.columns[:11]
)
print(rotation_df)

scores = pca.transform(md_x)
plt.figure(figsize=(8, 6))
plt.scatter(scores[:, 0], scores[:, 1], color='black')
plt.title("Principal components analysis of the fast food data set ( PC1 vs PC2)")
plt.xlabel("principal component 1")
plt.ylabel("principal component 2")
plt.grid()

for i, (com1, com2) in enumerate(pca.components_[:2].T):
    plt.arrow(0, 0, com1, com2, color='r', alpha=0.75, head_width=0.10, head_length=0.10),
    plt.text(com1*1.15, com2*1.15, mcdonalds.columns[i], color='r', ha='center', va='center')
plt.show()


#                                 STEP 5: EXTRACTING SEGMENTS

#                                       USING K-MEANS

np.random.seed(1234)
results = []
silhouette_val = []

for k in range(2,9):
    best_model = None
    best_score = -1
    for _ in range(10):
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=np.random.randint(0, 10000))
        kmeans.fit(md_x)
        score = silhouette_score(md_x, kmeans.labels_)
        if score > best_score:
            best_model = kmeans
            best_score = score

    results.append(best_model)
    silhouette_val.append(best_score)
best_k = np.argmax(silhouette_val) + 2
best_model = results[best_k - 2]
print(f"Best number of clusters: {best_k}")
labels = best_model.labels_

plt.figure(figsize=(8, 6))
plt.plot(range(2, 9), silhouette_val, marker='o')
plt.title('Performance of Clustering Solutions')
plt.xlabel('Number of Segments')
plt.ylabel('Silhouette Score')  # Useful to evaluate the quality of clustering.
plt.grid(True)
plt.show()


def bootstrap_kmeans_single(X, k, nrep=10):
    bootstrap_idx = np.random.choice(X.shape[0], X.shape[0], replace=True)
    X_bootstrap = X[bootstrap_idx, :]

    labels_list = [KMeans(n_clusters=k, n_init=10).fit(X_bootstrap).labels_ for _ in range(nrep)]
    ari_scores = []
    for i in range(nrep):
        for j in range(i + 1, nrep):
            ari_scores.append(adjusted_rand_score(labels_list[i], labels_list[j]))

    return np.mean(ari_scores)


def bootstrap_kmeans_parallel(X, n_clusters_range, nboot=100, nrep=10, n_jobs=-1):
    ari_results = {}

    for k in n_clusters_range:
        ari_scores = Parallel(n_jobs=n_jobs)(delayed(bootstrap_kmeans_single)(X, k, nrep) for _ in range(nboot))
        ari_results[k] = np.mean(ari_scores)

    return ari_results


n_clusters_range = range(2, 9)
nboot = 50
nrep = 10

ari_results = bootstrap_kmeans_parallel(md_x, n_clusters_range, nboot=nboot, nrep=nrep, n_jobs=-1)

plt.plot(list(ari_results.keys()), list(ari_results.values()), marker='o')
plt.xlabel('Number of Segments')
plt.ylabel('Adjusted Rand Index')
plt.grid(True)
plt.show()

kmeans_4 = KMeans(n_clusters=4, random_state=1234).fit(md_x)
labels_4 = kmeans_4.labels_

plt.hist(labels_4, bins=np.arange(5) - 0.5, rwidth=0.8)
plt.xlim(-0.5, 3.5)
plt.xlabel('Cluster Label')
plt.ylabel('Number of Points')
plt.title('Cluster Distribution (k=4)')
plt.grid(True)
plt.show()

#                                   Using Mixtures of Regression Models

frequency_table = mcdonalds['Like'].value_counts()
reversed_frequency_table = frequency_table[::1]
print(reversed_frequency_table)
Like_count = pd.DataFrame(reversed_frequency_table)
scaler = StandardScaler()
data_scaled = scaler.fit_transform(Like_count)

gmm_initial = GaussianMixture(n_components=11, random_state=1234)
gmm_initial.fit(data_scaled)

gmm_refitted = GaussianMixture(n_components=11, random_state=1234)
gmm_refitted.fit(data_scaled)
print("Convergence status:", gmm_refitted.converged_)
print("Number of iterations:", gmm_refitted.n_iter_)
print("Means of components:\n", gmm_refitted.means_)
print("Covariances of components:\n", gmm_refitted.covariances_)
print("Weights of components:\n", gmm_refitted.weights_)


#                                      STEP 6: PROFILING SEGMENTS


dataframe = pd.DataFrame(md_x)
kmeans = KMeans(n_clusters=4, random_state=1234)
dataframe['Cluster'] = kmeans.fit_predict(dataframe)

cluster_counts = dataframe['Cluster'].value_counts()
cc = pd.DataFrame(cluster_counts)
cluster_percentages = (cluster_counts / len(dataframe)) * 100
cp = pd.DataFrame(cluster_percentages)

print("Cluster Counts:\n", cluster_counts)
print("Cluster Percentages:\n", cluster_percentages)

cluster_means = dataframe.groupby('Cluster').mean()
cluster_means = cluster_means.reset_index()
print(cluster_means)

num_clusters = cluster_means['Cluster'].nunique()
fig, axes = plt.subplots(nrows=1, ncols=num_clusters, figsize=(14, 4), sharey=True)
fig.suptitle('Mean Feature Values by Cluster', fontsize=16)
color = ['#1576BB', '#458CCC', '#6AA6DA', '#95C1E6']

for i in range(num_clusters):
    cluster_data = cluster_means[cluster_means['Cluster'] == i].drop('Cluster', axis=1).melt(var_name='Feature', value_name='MeanValue')
    axes[i].barh(mcdonalds.columns[:11], cluster_data['MeanValue'], color=color[i])
    axes[i].set_title(f'Cluster {i}: {cc.iloc[i, 0].round(2)} ({cp.iloc[i, 0].round(2)}%)')
    axes[i].set_xlabel('Mean Value')

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()


#                                  STEP 7: DESCRIBING SEGMENTS


def color_fun(key):
    try:
        like_value = key[1]
        return {'color': like_colors.get(like_value, '#cccccc')}
    except (KeyError, ValueError) as e:
        print(f"Error in color_func with key {key}: {e}")
        return {'color': '#cccccc'}


def empty_labelizer(key):
    return ""


likes = mcdonalds['Like'][:100].unique()
like_colors = {
    likes[0]: '#ff9999',
    likes[1]: '#66b3ff',
    likes[2]: '#99ff99',
    likes[3]: '#ffcc99',
    likes[4]: '#ff6666',
    likes[5]: '#9999ff',
    likes[6]: '#66ff99',
    likes[7]: '#cc99ff',
    likes[8]: '#ffcc66',
    likes[9]: '#6666ff',
    likes[10]: '#ff99cc'
}
colors = ['#1576BB', '#458CCC', '#6AA6DA', '#95C1E6']
Like = pd.crosstab(dataframe['Cluster'], mcdonalds['Like'])
mosaic(Like.stack(), gap=0.10, title='Mosaic Plot of Clusters vs Like', properties=color_fun, labelizer=empty_labelizer)
patches = [mpatches.Patch(color=like_colors[like], label=like) for like in likes]
plt.legend(handles=patches, title="Like Values", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Segment Number')
plt.subplots_adjust(right=0.75)
plt.show()


def gender_color_fun(key):
    try:
        gender_value = key[1]
        return {'color': gender_colors.get(gender_value, '#cccccc')}
    except (KeyError, ValueError) as e:
        print(f"Error in color_func with key {key}: {e}")
        return {'color': '#cccccc'}


gender_colors = {
    'Male': '#000991',
    'Female': '#EC3699'
}
Gender = pd.crosstab(dataframe['Cluster'], mcdonalds['Gender'])
mosaic(Gender.stack(), gap=0.10, title='Mosaic Plot of Clusters vs Gender', properties=gender_color_fun, labelizer=empty_labelizer)
patches = [mpatches.Patch(color=gender_colors[gender], label=gender) for gender in gender_colors]
plt.legend(handles=patches, title="Gender", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.xlabel('Segment Number')
plt.subplots_adjust(right=0.75)
plt.show()


#                                  STEP 8: SELECTING THE TARGET SEGMENTS


like_encoder = LabelEncoder()
visit_encoder = LabelEncoder()

mcdonalds['Like_numeric'] = like_encoder.fit_transform(mcdonalds['Like'])
mcdonalds['Visit_numeric'] = visit_encoder.fit_transform(mcdonalds['VisitFrequency'])
dataframe = pd.DataFrame(md_x)
kmeans = KMeans(n_clusters=4, random_state=1234)
mcdonalds['Cluster'] = kmeans.fit_predict(dataframe)
plt.figure(figsize=(8, 6))

for cluster in mcdonalds['Cluster'].unique():
    cluster_data = mcdonalds[mcdonalds['Cluster'] == cluster]
    plt.scatter(cluster_data['Visit_numeric'], cluster_data['Like_numeric'],
                label=f'Cluster {cluster}',
                alpha=0.6, edgecolors='w', s=100)

plt.xlabel('Visit (Numeric)')
plt.ylabel('Like (Numeric)')
plt.title('Visit Frequency vs Like (Clusters)')
plt.legend(title='Clusters', loc='upper left', bbox_to_anchor=(1, 1))
plt.subplots_adjust(right=0.75)
plt.grid(True)
plt.show()


