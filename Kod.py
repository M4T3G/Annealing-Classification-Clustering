import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

# Veri setini okuma
df = pd.read_excel('tumveri.xlsx')

# Veri seti hakkında genel bilgiler
print("Veri Seti Boyutu:", df.shape)
print("\nVeri Seti Özellikleri:")
print(df.info())
print("\nİlk 5 Satır:")
print(df.head())

# Eksik veri analizi
print("\nEksik Veri Analizi:")
print(df.isnull().sum())

# Sayısal sütunları seçme (object tipinde olmayanlar)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# Aykırı değer analizi için box plot (sadece sayısal sütunlar)
plt.figure(figsize=(15, 6))
df[numeric_cols].boxplot()
plt.xticks(rotation=45)
plt.title('Aykırı Değer Analizi (Sayısal Özellikler)')
plt.tight_layout()
plt.savefig('aykiri_deger_analizi.png')
plt.close()

# Korelasyon analizi (sadece sayısal sütunlar)
if len(numeric_cols) > 1:  # En az iki sayısal sütun varsa
    plt.figure(figsize=(12, 8))
    sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
    plt.title('Korelasyon Matrisi (Sayısal Özellikler)')
    plt.tight_layout()
    plt.savefig('korelasyon_matrisi.png')
    plt.close()
else:
    print("\nKorelasyon analizi için yeterli sayısal özellik bulunamadı.")

# Sınıf dağılımı analizi
if 'class' in df.columns:  # Sınıf sütununun adını kontrol et
    print("\nSınıf Dağılımı:")
    print(df['class'].value_counts())
    
    # Sınıf dağılımı görselleştirme
    plt.figure(figsize=(8, 6))
    df['class'].value_counts().plot(kind='bar')
    plt.title('Sınıf Dağılımı')
    plt.xlabel('Sınıf')
    plt.ylabel('Örnek Sayısı')
    plt.tight_layout()
    plt.savefig('sinif_dagilimi.png')
    plt.close()

# Özellik önem dereceleri analizi
if 'class' in df.columns:
    # Sadece sayısal sütunları kullanarak X'i oluştur
    X = df[numeric_cols].drop('class', axis=1, errors='ignore')
    y = df['class']
    
    if not X.empty:  # Eğer sayısal özellikler varsa
        # Veriyi ölçeklendirme
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Random Forest ile özellik önem dereceleri
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_scaled, y)
        
        feature_importance = pd.DataFrame({
            'Özellik': X.columns,
            'Önem Derecesi': rf.feature_importances_
        }).sort_values('Önem Derecesi', ascending=False)
        
        print("\nÖzellik Önem Dereceleri (Sayısal Özellikler):")
        print(feature_importance)
        
        # Özellik önem dereceleri görselleştirme
        plt.figure(figsize=(10, 6))
        sns.barplot(x='Önem Derecesi', y='Özellik', data=feature_importance)
        plt.title('Özellik Önem Dereceleri (Sayısal Özellikler)')
        plt.tight_layout()
        plt.savefig('ozellik_onem_dereceleri.png')
        plt.close()
    else:
        print("\nÖzellik önem dereceleri analizi için yeterli sayısal özellik bulunamadı.")


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns

# Veri setini okuma
df = pd.read_excel('tumveri.xlsx')

# Veriyi hazırlama
X = df.drop('class', axis=1)
y = df['class']

# Veriyi ölçeklendirme
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Eğitim ve test verilerini ayırma
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Sınıflandırma modelleri
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', random_state=42),
    'KNN': KNeighborsClassifier(n_neighbors=5)
}

# Sınıflandırma sonuçları
print("Sınıflandırma Sonuçları:")
print("-" * 50)

for name, clf in classifiers.items():
    print(f"\n{name} Sınıflandırıcı:")
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    
    # Karışıklık matrisi
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} Karışıklık Matrisi')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.ylabel('Gerçek Sınıf')
    plt.tight_layout()
    plt.savefig(f'{name}_karisiklik_matrisi.png')
    plt.close()
    
    # Sınıflandırma raporu
    print("\nSınıflandırma Raporu:")
    print(classification_report(y_test, y_pred))

# Kümeleme analizi
print("\nKümeleme Analizi:")
print("-" * 50)

# K-means kümeleme
kmeans = KMeans(n_clusters=len(y.unique()), random_state=42)
kmeans_labels = kmeans.fit_predict(X_scaled)
kmeans_silhouette = silhouette_score(X_scaled, kmeans_labels)
print(f"\nK-means Silhouette Skoru: {kmeans_silhouette:.3f}")

# DBSCAN kümeleme
dbscan = DBSCAN(eps=0.5, min_samples=5)
dbscan_labels = dbscan.fit_predict(X_scaled)
if len(np.unique(dbscan_labels)) > 1:  # En az 2 küme varsa
    dbscan_silhouette = silhouette_score(X_scaled, dbscan_labels)
    print(f"DBSCAN Silhouette Skoru: {dbscan_silhouette:.3f}")

# Hiyerarşik kümeleme
hierarchical = AgglomerativeClustering(n_clusters=len(y.unique()))
hierarchical_labels = hierarchical.fit_predict(X_scaled)
hierarchical_silhouette = silhouette_score(X_scaled, hierarchical_labels)
print(f"Hiyerarşik Kümeleme Silhouette Skoru: {hierarchical_silhouette:.3f}")

# Kümeleme sonuçlarını görselleştirme
plt.figure(figsize=(15, 5))

# K-means görselleştirme
plt.subplot(131)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=kmeans_labels, cmap='viridis')
plt.title('K-means Kümeleme')

# DBSCAN görselleştirme
plt.subplot(132)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=dbscan_labels, cmap='viridis')
plt.title('DBSCAN Kümeleme')

# Hiyerarşik kümeleme görselleştirme
plt.subplot(133)
plt.scatter(X_scaled[:, 0], X_scaled[:, 1], c=hierarchical_labels, cmap='viridis')
plt.title('Hiyerarşik Kümeleme')

plt.tight_layout()
plt.savefig('kumeleme_sonuclari.png')
plt.close() 