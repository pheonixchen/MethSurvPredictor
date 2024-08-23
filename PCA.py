import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import joblib


script_path = os.path.abspath(__file__)
script_dir = os.path.dirname(script_path)
os.chdir(script_dir)


file_path = 'TCGA-LGG.methylation450.tsv'  
df = pd.read_csv(file_path, sep='\t', index_col=0)



df.dropna(inplace=True)


scaler = StandardScaler()
scaled_data = scaler.fit_transform(df.T)  


pca = PCA(n_components=50)  
principal_components = pca.fit_transform(scaled_data)


pca_model_path = 'pca_model.pkl'
joblib.dump(pca, pca_model_path)
print(f"PCA模型已保存为 {pca_model_path}")


loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(pca.n_components_)], index=df.index)
loadings.to_csv('pca_loadings.csv')
print("主成分载荷矩阵已保存为 pca_loadings.csv")


sample_ids = df.columns
principal_df = pd.DataFrame(data=principal_components, columns=[f'Principal Component {i+1}' for i in range(50)], index=sample_ids)


fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(principal_df['Principal Component 1'], principal_df['Principal Component 2'], principal_df['Principal Component 3'])

for i, sample_id in enumerate(sample_ids):
    ax.text(principal_df['Principal Component 1'][i], principal_df['Principal Component 2'][i], principal_df['Principal Component 3'][i], sample_id)

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D PCA of Methylation Data')
plt.show()


output_file_path = 'pca_principal_components.csv'
principal_df.to_csv(output_file_path)


explained_variance = pca.explained_variance_ratio_
print(f"Explained variance by each component: {explained_variance}")

print(f"50个主成分已保存为 {output_file_path}")
