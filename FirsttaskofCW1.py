# Import libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pandas import DataFrame
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from yellowbrick.cluster import KElbowVisualizer, SilhouetteVisualizer
import scipy.cluster.hierarchy as shc
from sklearn.cluster import AgglomerativeClustering

# Import dataset and use the X to instead of data for our dataset
data = pd.read_csv('C:/Users/msham/PycharmProjects/DataMiningCoursework/CW1_COVID_world_data.csv')
X = data

# Understand number of columns and rows
print(X.shape)
print(X.head())

# Understanding types of data stored in our dataset
print(X.dtypes)

# Understand the number of missing values in each column by the number of them and the percentage
print(X.isna().sum())
print(X.isna().sum() / len(X) * 100)

# Drop first and last row
X.drop([0, 226], inplace=True)
print(X)

# Understand data in the table
print(X.describe())
# Finding null values in population
null_columns = X.columns[X.isnull().any()]
print(X[X["Population"].isnull()][null_columns])


# Total deaths and population are objects therefor we should change them to numbers
# First we need to get rid of commas in them
X['Population'] = X['Population'].str.replace(',', '')
X['Total Deaths'] = X['Total Deaths'].str.replace(',', '')

# then we use astype to change type and for that we need all the values not to be nan, so we put
# zeros, but then we change them to null, so we could perform our analysis
X["Total Deaths"].fillna(0, inplace=True)
X["Population"].fillna(0, inplace=True)
X[["Total Deaths", "Population"]] = X[["Total Deaths", "Population"]].astype(int)
print(X.dtypes)
X[['Total Deaths']] = X[['Total Deaths']].replace(0, np.nan)

# Drop countries with zero population from last part
for x in X.index:
    if X['Population'][x]==0:
        X = X.drop(x)
# we should check to not create null values more than before
print(X.isna().sum())




# replacing na values in New cases and New deaths and new recovered with zeros
X["New Cases"].fillna(0, inplace=True)
X["New Deaths"].fillna(0, inplace=True)
X["New Revovered"].fillna(0, inplace=True)

# replace other null values with mean of each column
X.fillna(X.mean(), inplace=True)
print(X.isna().sum())
print(X.corr())


# Put country names out of the table for standardizing the data and put it in z dataset and
# also resting the index of country names for merging them to pca outcome after pca analysis
data_before_normalisation = X.iloc[:, 1:]
country_names = X.iloc[:, :1]
country_names.reset_index(drop=True, inplace=True)
# Now our data is normalized and ready to process the first thing is feature extraction and
# based on that and the columns
# it is obvious that we need to drop total cases and total deaths and total tests because we have them
# in total per million, and it is unnecessary to have them. However, new deaths and new recovered and new deaths
# involves more than 80 percent of zeros, and we should drop them. Having only recovered cases and critical cases
# is not helpful, so we can change it to recovered and critical cases per million then having also
# active cases when it is the subtraction of recovered and total doesn't mean a lot.
# So the best way to summarise data is to have population, Total death per million, Total case per million
# Total test per million, active cases per million, critical cases per million.

conditions = [
    (data_before_normalisation['Population'] < 1000000),
    (data_before_normalisation['Population'] >= 1000000)
    ]

# create a list of the values we want to assign for each condition
value1 = [data_before_normalisation['Active Cases']/round(1000000/(data_before_normalisation['Population'])),
         data_before_normalisation['Active Cases'] / round((data_before_normalisation['Population'])/1000000  )]
value2 = [data_before_normalisation['Serious, Critical Cases']/round(1000000/(data_before_normalisation['Population'])),
         data_before_normalisation['Serious, Critical Cases'] / round((data_before_normalisation['Population'])/1000000  )]

# create a new column and use np.select to assign values to it using our lists as arguments
data_before_normalisation['Active Cases/ 1M Pop'] = np.select(conditions, value1)
data_before_normalisation['Serious, Critical Cases/ 1M Pop'] = np.select(conditions, value2)

# display updated DataFrame
data_before_normalisation.head()
data_before_normalisation.drop(["Total Cases", "New Cases", "Total Deaths", "New Deaths", "Active Cases"
                                ,"Serious, Critical Cases","Total Recovered", "New Revovered","Total Tests"]
                               ,axis=1,inplace=True)
'''data_before_normalisation.drop(["New Revovered", "New Cases", "New Deaths", "Active Cases"],axis=1,inplace=True)'''
print(data_before_normalisation.head())
print(data_before_normalisation.shape)

# Save the data before normalisation into a csv file
data_before_normalisation.to_csv('CW1_firstpart_databeforenormalisation.csv', index=False)
# standardizing the data
scaler = preprocessing.Normalizer()
data_scaled = scaler.fit_transform(data_before_normalisation)
data_after_normalisation = pd.DataFrame(data_scaled,columns = data_before_normalisation.columns)
print(data_after_normalisation.head())
print(data_after_normalisation.shape)
print(data_after_normalisation.columns)


# create a PCA instance
pca = PCA(n_components=5)
principalComponents = pca.fit_transform(data_after_normalisation)

# Plot scree plot
PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'o-', linewidth=2, color='blue')
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Variance Explained')
plt.show()
plt.savefig("course3workpart1_EMG_YB.png", dpi=1080, format='png')

# plot with the explained variances
features = range(pca.n_components_)
plt.bar(features, pca.explained_variance_ratio_, color='black')
plt.xlabel('PCA features')
plt.ylabel('variance %')
plt.xticks(features)
plt.savefig("course2workpart1_EMG_YB.png", dpi=1080, format='png')
plt.show()

# Plot scree plot with another code and with better resoulotion
plt.figure(figsize=(10, 10))
plt.title('Scree Plot 2')
var = np.round(pca.explained_variance_ratio_ * 100, decimals=1)
lbls = [str(x) for x in range(1, len(var) + 1)]
plt.bar(x=range(1, len(var) + 1), height=var, tick_label=lbls)
plt.show()

# Understand the importance and impact of each feature on each PCA components
print(pca.explained_variance_ratio_)
print(abs(pca.components_))

# Change back pca to dataset
PCA_components = pd.DataFrame(principalComponents)
print(PCA_components.head())
print(PCA_components.shape)
PCA_outcome = PCA_components
print(PCA_outcome.shape)

print(country_names.shape)
print(country_names.head())

# Put the name of country back in the first row

data_ready = pd.merge(country_names, PCA_components, left_index=True, right_index=True)

print(data_ready.head())
print(data_ready.shape)
data_for_cluster = data_ready.iloc[:, :3]
print(data_for_cluster.shape)

# doing the k means algo
wcss = []
# fitting multiple k-means algorithms
for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++').fit(data_for_cluster.iloc[:, 1:])
    wcss.append(kmeans.inertia_)

print(wcss)

# Elbow method with Yellowbrick Visualiser
visualizer = KElbowVisualizer(kmeans, k=(2, 11))
visualizer.fit(data_for_cluster.iloc[:, 1:])
visualizer.show()
visualizer.show(outpath="coursework1part1_EMG_YB.png")

# Silhouette score for finding number of clusters

for i in range(2, 11):
    kmeans = KMeans(n_clusters=i, init='k-means++')
    visualizer = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    visualizer.fit(data_for_cluster.iloc[:, 1:])
    visualizer.show()

# Creating 4 clusters
kmeans = KMeans(n_clusters=4, init='k-means++', max_iter=300, n_init=10, random_state=0)
y_kmeans = kmeans.fit_predict(data_for_cluster.iloc[:, 1:])
centers = np.array(kmeans.cluster_centers_)
# count of points in each of the above-formed clusters
frame = pd.DataFrame(data_for_cluster.iloc[:, 1:])
frame['Cluster'] = y_kmeans
frame['Cluster'].value_counts()
print(frame['Cluster'].value_counts())

# visualising the clusters
# colors for plotting
colors = ['red', 'magenta','blue','green']

# assign a color to each feature (note that we are using features as target)
features_colors = [colors[y_kmeans[i]] for i in range(len(data_for_cluster.iloc[:, 1:]))]
T = data_for_cluster.iloc[:, 1:]

# plot the PCA cluster components
plt.scatter(T[0], T[1], c=features_colors, marker='o', alpha=0.4)
plt.scatter(centers[:, 0], centers[:, 1], marker='x', s=100, linewidths=3, c=colors)
plt.text(x=data_for_cluster.iloc[:, 1][data_for_cluster.iloc[:, 0] == 'Iran']+0.001,
         y=data_for_cluster.iloc[:, 2][data_for_cluster.iloc[:, 0] == 'Iran']+0.001,
         s="Iran", fontdict=dict(color='black', size=10, bbox=dict(facecolor='yellow', alpha=0.4)))
plt.text(x=data_for_cluster.iloc[:, 1][data_for_cluster.iloc[:, 0] == 'USA']+0.001,
         y=data_for_cluster.iloc[:, 2][data_for_cluster.iloc[:, 0] == 'USA']+0.001,
         s="USA", fontdict=dict(color='black', size=10, bbox=dict(facecolor='yellow', alpha=0.4)))
plt.text(x=data_for_cluster.iloc[:, 1][data_for_cluster.iloc[:, 0] == 'UK']+0.001,
         y=data_for_cluster.iloc[:, 2][data_for_cluster.iloc[:, 0] == 'UK']+0.001,
         s="UK", fontdict=dict(color='black', size=10, bbox=dict(facecolor='yellow', alpha=0.4)))

plt.title('K-means Clustering with 2 dimensions')
# store the values of PCA component in variable: for easy  writing
xvector = pca.components_[1] * max(T[0])
yvector = pca.components_[2] * max(T[1])
columns = data_after_normalisation.columns
# plot the 'name of individual features' along with vector length
for i in range(len(columns)):
    # plot arrows
    plt.arrow(0, 0, xvector[i], yvector[i], color='blue', width=0.005,head_width=0.04, alpha=0.5)
    # plot name of features
    plt.text(xvector[i], yvector[i], list(columns)[i], color='black',size=10, alpha=0.7)

plt.title('K means Clusters Visualization')
plt.show()
plt.savefig('courseworkpart1_visualization_kmeans.png', dpi=1080, format='png')

# Hierarchical clustering part
# Dendrogram creation
plt.figure()
plt.title("Countries Dendograms")
dend = shc.dendrogram(shc.linkage(data_for_cluster.iloc[:, 1:], method='ward'))
plt.savefig('courseworkpart1_dendograms.png', dpi=1080, format='png')
plt.show()

# Hierarchical clustering based on the number of dendrogram previous part
cluster = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
hierarchical_cluster = cluster.fit_predict(data_for_cluster.iloc[:, 1:])
print(hierarchical_cluster)

# count of points in each of the above-formed clusters
frame = pd.DataFrame(data_for_cluster.iloc[:, 1:])
frame['Hierarchical Clusters'] = hierarchical_cluster
frame['Hierarchical Clusters'].value_counts()
print(frame['Hierarchical Clusters'].value_counts())

# Visualizing part

features_colors = [colors[hierarchical_cluster[i]] for i in range(len(data_for_cluster.iloc[:, 1:]))]
T = data_for_cluster.iloc[:, 1:]

# plot the PCA cluster
plt.figure()
sns.scatterplot(data=data_for_cluster, x=data_for_cluster.iloc[:, 1], y=data_for_cluster.iloc[:, 2], c=features_colors,
                marker='o', alpha=0.3)

plt.title('Hierarchical Clusters Visualization without country names')
plt.show()
plt.savefig('courseworkpart1_visualization_hierarchicalwithoutcountrynames.png', dpi=1080, format='png')
# Visualizing part
features_colors = [colors[hierarchical_cluster[i]] for i in range(len(data_for_cluster.iloc[:, 1:]))]
T = data_for_cluster.iloc[:, 1:]

# plot the PCA cluster
plt.figure()
sns.scatterplot(data=data_for_cluster, x=data_for_cluster.iloc[:, 1], y=data_for_cluster.iloc[:, 2], c=features_colors,
                marker='o', alpha=0.3)
for i in range(data_for_cluster.shape[0]):
    plt.text(x=data_for_cluster.iloc[:, 1][i]+0.001, y=data_for_cluster.iloc[:, 2][i]+0.001,
             s=data_for_cluster.iloc[:, 0][i],fontdict=dict(color='black',size=10,
                                                            bbox=dict(facecolor='yellow',alpha=0.4)))

plt.title('Hierarchical Clusters Visualization with country names')
plt.show()
plt.savefig('courseworkpart1_visualization_hierarchicalwithcountrynames.png', dpi=1080, format='png')
