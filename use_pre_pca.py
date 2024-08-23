import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
script_path=os.path.abspath(__file__)
script_dir=os.path.dirname(script_path)
os.chdir(script_dir)

pca_model_path = 'pca_model.pkl'
loaded_pca = joblib.load(pca_model_path)



file_path = 'TCGA-LGG.methylation450.tsv'  
new_data = pd.read_csv(file_path, sep='\t', index_col=0)



new_data.dropna(inplace=True)


scaler = StandardScaler()
scaled_new_data = scaler.fit_transform(new_data.T)  


new_principal_components = loaded_pca.transform(scaled_new_data)


sample_ids = new_data.columns
new_principal_df = pd.DataFrame(data=new_principal_components, columns=[f'Principal Component {i+1}' for i in range(loaded_pca.n_components_)], index=sample_ids)


print(new_principal_df)

output_file_path = 'pca_principal_components.csv'
new_principal_df.to_csv(output_file_path)