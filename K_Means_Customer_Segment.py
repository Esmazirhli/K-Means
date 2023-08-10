# Gerekli Import İşlemleri

# pip install yellowbrick
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import StandardScaler
pd.set_option("display.width",None)
pd.set_option("display.max_columns",None)
pd.set_option("display.max_rows",None)

# Veri Seti ve Birleştirme İşlemleri (Segmentasyon işlemi yapılacağı için iki veri seti birleştirilip segmentasyon alanı siliniyor)
df1 = pd.read_csv("datasets/otomotive_train-set.csv")
df2 = pd.read_csv("datasets/otomotive_test_set.csv")

df = pd.concat([df1, df2], ignore_index=True)


# K-Means algoritmasında kullanılmayacak alanların silinme işlemi
df = df.drop(["CustomerID","Segmentation"],axis=1)


# Veri setinin betimsel istatistikleri
df.info()
df.isnull().sum()
df.shape
df.head()
df.describe().T


# Boş değerlerin "Unknown" ile doldurulması
df = df.fillna("Unknown")

# Veri Setinin Ön Hazırlık Süreci

# Married alanı için yaş ortalamasına bakılması
result = df.groupby(["Gender", "Married"]).agg({"Age": "mean"})

# Married alanının yaş ve cinsiyete göre doldurulması
df.loc[(df["Gender"] == "Female") & (df["Age"] <= 33)  & (df["Married"] == "Unknown"),"Married"] = "No"
df.loc[(df["Gender"] == "Female") & (df["Age"] >= 34)  & (df["Married"] == "Unknown"),"Married"] = "Yes"

df.loc[(df["Gender"] == "Male") & (df["Age"] <= 30)  & (df["Married"] == "Unknown"),"Married"] = "No"
df.loc[(df["Gender"] == "Male") & (df["Age"] >= 31)  & (df["Married"] == "Unknown"),"Married"] = "Yes"


# Diğer boş alanların en çok tekrar eden değere göre doldurulması
df["WorkExperience"] = df["WorkExperience"].replace("Unknown", df["WorkExperience"].mode()[0])
df["FamilySize"] = df["FamilySize"].replace("Unknown", df["FamilySize"].mode()[0])
df["Graduated"] = df["Graduated"].replace("Unknown", df["Graduated"].mode()[0])
df["Profession"] = df["Profession"].replace("Unknown", df["Profession"].mode()[0])
df["Category"] = df["Category"].replace("Unknown", df["Category"].mode()[0])


# Kategorik değişkenlerin factorize işlemi ile sayısal değer alma işlemi
df["Gender"] = pd.factorize(df["Gender"])[0]
df["Married"] = pd.factorize(df["Married"])[0]
df["Graduated"] = pd.factorize(df["Graduated"])[0]
df["Profession"] = pd.factorize(df["Profession"])[0]
df["SpendingScore"] = pd.factorize(df["SpendingScore"])[0]
df["Category"] = pd.factorize(df["Category"])[0]

################################
# K-Means
################################

# K-Means algoritması için tablodaki tüm alanlar için standartlaştırılma yapılması
sc = MinMaxScaler((0, 1))
df = sc.fit_transform(df)
df[0:5]

# K-Means modelinin kurulması ve oluşan parametlerin görüntülenmesi
kmeans = KMeans(n_clusters=4, random_state=17).fit(df)
kmeans.get_params()

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
kmeans.inertia_
# SSD değeri
# 5849.435460287319

##############################
# Optimum Küme Sayısı
################################

# Elbow yöntemi de kullanılarak veri seti için en iyi küme sayısının belirlenmesi

kmeans = KMeans()
ssd = []
K = range(1, 30)

for k in K:
    kmeans = KMeans(n_clusters=k).fit(df)
    ssd.append(kmeans.inertia_)

plt.plot(K, ssd, "bx-")
plt.xlabel("Farklı K Değerlerine Karşılık SSE/SSR/SSD")
plt.title("Optimum Küme sayısı için Elbow Yöntemi")
plt.show(block = True)

kmeans = KMeans()
elbow = KElbowVisualizer(kmeans, k=(2, 10))
elbow.fit(df)
elbow.show()

elbow.elbow_value_
# 5

################################
# Final Cluster
################################
elbow.elbow_value_
kmeans = KMeans(n_clusters=5).fit(df)

kmeans.n_clusters
kmeans.cluster_centers_
kmeans.labels_
df[0:5]

clusters_kmeans = kmeans.labels_

df = pd.concat([df1, df2], ignore_index=True)

df = df.drop(["Segmentation"],axis=1)

# Yeni kümeleri dataframe'e ekleme
df["cluster"] = clusters_kmeans

df["cluster"] = df["cluster"] + 1

# Yaş aralıklarını düzenleme
age_labels = ['0-4', '5-9', '10-14', '15-19', '20-24', '25-29',
                  '30-34', '35-39', '40-44', '45-49', '50-54', '55-59',
                  '60-64', '65-69', '70-74', '75-79', '80+']

bin_edges = [0, 4, 9, 14, 19, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, float('inf')]
df['Age_Arrival_Text'] = pd.cut(df['Age'], bins=bin_edges, labels=age_labels)

df.head()

# Gruplarda cinsiyet ve evlilik durumuna göre yaş iş deneyimi ve aile büyüklüğünü sorgulama
result = df.groupby(['cluster','Gender','Married']).agg(mean_age=('Age', 'mean'),
                                                     WorkExp=('WorkExperience', 'mean'),
                                                    Family=('FamilySize', 'mean'),
                                                                        Count=('cluster', 'count'))

#                          mean_age   WorkExp    Family  Count
# cluster Gender Married
# 1       Female No       37.608076  3.540121  2.316931   1263
#         Male   No       35.931322  3.272306  2.469880    961
# 2       Male   Yes      51.207792  2.324913  2.682572   2541
# 3       Female Yes      52.012148  2.405569  2.593269   1811
# 4       Female Yes      51.314706  2.553043  2.856269    680
#         Male   Yes      51.735632  2.075075  3.033852   1131
# 5       Female No       28.386318  2.935080  3.475930    994
#         Male   No       26.170819  2.306706  3.781716   1124

# Gruplar incelendiğinde 2-3-4 numaralı kümelerin birleştirilebileceği görüntüleniyor bu nedenle replace işlemi ile tek küme haline getirilmesi
df["cluster"] = df["cluster"].replace({1: 2, 2: 3, 4: 3, 3: 3, 5: 1})