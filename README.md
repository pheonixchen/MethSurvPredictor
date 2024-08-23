# MethSurvPredictor
[![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Model-blue)](https://huggingface.co/csycsycsy/MethSurvPredictor)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13366876.svg)](https://doi.org/10.5281/zenodo.13366876)
## Model on Hugging Face
<a href="https://huggingface.co/csycsycsy/MethSurvPredictor">
    <img src="https://th.bing.com/th/id/OIP.Bz8NJZhyjcCKXPEpSPQD4wHaE8?rs=1&pid=ImgDetMain" alt="Hugging Face" width="150"/>
</a>

MethSurvPredictor is a robust machine learning model designed to predict the prognosis of Low-Grade Glioma (LGG) patients by analyzing methylation data, utilizing a Multi-Layer Perceptron (MLP) architecture to deliver accurate survival predictions.
You can find and use the MethSurvPredictor model directly on [Hugging Face](https://huggingface.co/csycsycsy/MethSurvPredictor).

---
license: mit
---
MethSurvPredictor: Predicting Prognosis for LGG Patients Using Methylation Data
MethSurvPredictor is a machine learning model designed to predict the prognosis of Low-Grade Glioma (LGG) patients through the analysis of methylation data. The model utilizes a Multi-Layer Perceptron (MLP) architecture to deliver precise prognostic predictions, making it a valuable tool for both researchers and clinicians.

Model Overview
Authors:

First Author & Corresponding Author: Shuaiyu Chen
Co-First Authors: Jingyu Chen, Yuheng Guan
Architecture: Multi-Layer Perceptron (MLP)

The model includes multiple layers, each with dropout (0.5) to prevent overfitting. Early stopping was applied during training to avoid overfitting.
Data Processing: Due to the complexity and volume of methylation data, Principal Component Analysis (PCA) was performed, selecting 50 principal components as input for the model.
Training Strategy: The model was trained on the TCGA-GDC LGG methylation dataset, with a random split into training and validation sets. GPU acceleration is supported to improve training efficiency.
Dependencies:

Libraries: pandas, numpy, matplotlib, scikit-learn, scipy, torch
Implemented in PyTorch, compatible with both CPU and GPU environments.
Performance Metrics
R²: ~0.5
Spearman Correlation: 0.72
P-value: 1.9e-05
MethSurvPredictor outperforms traditional Kaplan-Meier (KM) models by providing more accurate numerical predictions, making it a crucial tool for clinical decision-making.

Application and Usage
Implementation: The model can be easily implemented using PyTorch, loading the best_model.pth file for the best-trained weights. Pre-trained PCA components (pca_model.pkl) ensure consistent data preprocessing.
Output Interpretation: The model generates a continuous prognostic score, which can be interpreted to provide nuanced insights into patient outcomes.
Model Extensions and Customization
Scalability: The model is designed to be easily extended to other cancer types or additional data features, such as miRNA, CNV, and SNV, to enhance predictive accuracy.
Hyperparameter Tuning: The hyperparameters.py script can be used to fine-tune model parameters, optimizing performance based on specific datasets.
Model Explainability
Interpreting the Model’s Decisions: Understanding the model’s decision-making process is crucial, especially for clinical applications. The radar charts generated by webplot.py help visualize feature importance, providing insights into which factors contribute most to the model’s predictions.
Data Quality and Ethics
Data Cleaning and Quality Control: Key steps in data cleaning and quality control should be considered, such as handling missing values, outlier detection, and ensuring data balance, to maintain the integrity of the model’s predictions.
Ethical Considerations: Handling patient data involves ethical responsibilities. Privacy and data security should be prioritized, adhering to ethical standards in research involving human subjects.
Limitations
Feature Limitation: The model currently relies solely on methylation data, which may limit its generalizability.
Demographic Bias: The model is primarily trained on data from White patients in TCGA, which may affect its applicability across diverse populations.
Project Files
Training_Data_Generation.R: Generates the data.csv file for training.
webplot.py: Plots radar charts to visualize model performance or feature importance.
use_pre_pca.py: Applies the pre-trained PCA model.
PCA.py: Performs PCA on methylation data.
methane.py: Trains the MLP model.
hyperparameters.py: Optimizes model hyperparameters.
best_model.pth: Contains the best model weights.
pca_model.pkl: Contains the pre-trained PCA model.
pca_principal_components.csv: Stores the PCA principal components.
data.csv: Contains the training data.
Keywords
LGG, Prognosis, Methylation, Machine Learning, TCGA

Visualization and Results

Model Training and Performance Overview (Figure 1)

Model Architecture: Figure 1a illustrates the architecture of the MethSurvPredictor model, which is a Multi-Layer Perceptron (MLP) with multiple interconnected layers. Each node represents a neuron, and the lines between nodes depict the synaptic weights learned during training. This architecture visualization helps in understanding the complexity of the model and how it captures intricate patterns within the methylation data.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66c5cec3f720e602d299597a/ZaxPSpvLG8j3iM_06T9HV.png)

Training and Test Loss: Figure 1b displays the training and test loss over 10,000 epochs. The blue line represents the training loss, the green line represents the test loss, and the dashed lines indicate their moving averages. The steady decrease in loss over time suggests that the model is effectively learning and generalizing from the data. The convergence of these losses indicates that the model achieves optimal performance while avoiding overfitting.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66c5cec3f720e602d299597a/dcwVfVFSs0CLTH9VWZjyZ.png)

R² Score Progression: Figure 1c shows the progression of the R² score during training. The R² score gradually increases and stabilizes around 0.5, indicating a moderate correlation between the model’s predictions and the actual data. This demonstrates that the model successfully captures significant patterns in the input data and explains a substantial portion of the variance in the output.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66c5cec3f720e602d299597a/5TVVPU6P9kGdVvgG03VOk.png)

Predictions vs. Actual Values: Figure 1d presents a scatter plot comparing the predicted overall survival (OS) times with the actual OS times. The results show a strong correlation coefficient of r=0.72 (p=1.9e-05), indicating that the model is highly accurate in its predictions. Although there is some variance, most points lie close to the ideal fit line (red dashed line), confirming the model’s overall accuracy.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66c5cec3f720e602d299597a/YFgA2hBYS3wMp0hZjiofw.png)

Actual vs. Predicted Values Over Time: Figure 1e is a 3D plot that compares predicted OS times with actual OS times across different sample indices. The peaks and troughs represent variations in survival times, with the blue line indicating the predicted values. The close alignment of these peaks and troughs with the actual data highlights the model's ability to track changes in patient survival outcomes over time.

Model Weight Analysis and 

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66c5cec3f720e602d299597a/y9eX0zjCvvQDw4ElU0BWs.png)

Feature Importance (Figure 2)
Layer 5 Weight Distribution and Heatmap: Figure 2a shows the weight distribution in Layer 5, where weights are mostly concentrated in the lower range, with some extending to higher values. This indicates that while most connections are relatively weak, a few strong connections may play a crucial role in the model’s decision-making process. Figure 2b, the weight heatmap of Layer 5, highlights the intensity of these connections, with brighter areas (yellow lines) indicating stronger connections, pointing to the most influential features in the model’s predictions.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66c5cec3f720e602d299597a/kUz2xr0AQ6g8Njnbjwjlt.png)
![image/png](https://cdn-uploads.huggingface.co/production/uploads/66c5cec3f720e602d299597a/QyjGB7ix-tfN9FXoGW7zF.png)

Layer 3 Weight Distribution and Heatmap: Figure 2c shows a more balanced weight distribution in Layer 3, suggesting a more even contribution of various connections in this layer. The heatmap in Figure 2d reveals a more diffuse pattern of connections, with fewer distinct strong connections, indicating that Layer 3 is likely involved in refining the feature representations passed down from previous layers.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66c5cec3f720e602d299597a/PQjFx--WhB9DRjGUCx8Nb.png)
![image/png](https://cdn-uploads.huggingface.co/production/uploads/66c5cec3f720e602d299597a/MOkWrmuzEEggtcYw1SPV5.png)

Layer 1 Weight Distribution and Heatmap: Figure 2e depicts the weight distribution in Layer 1, which shows a near-normal distribution centered around zero. This balanced distribution is typical for initial layers in a neural network, where the model begins learning basic patterns from the input data. Figure 2f’s heatmap shows a broad distribution of weights with no single feature dominating, indicating that Layer 1 is learning a wide variety of basic patterns from the data.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66c5cec3f720e602d299597a/7Q6w9W_pGeMGrQ8cm1CC0.png)
![image/png](https://cdn-uploads.huggingface.co/production/uploads/66c5cec3f720e602d299597a/2FWYGjrzNGfKtMZn1SWCG.png)

Feature Importance in the First Layer: Figure 2g ranks the importance of different features in the first layer based on the absolute weights of the connections. The top-ranked features stand out significantly from the rest, suggesting that certain methylation sites or components play a critical role in determining the survival outcomes of LGG patients.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66c5cec3f720e602d299597a/86_oL4b-rPkyWGZINCBBn.png)

Prediction Error Distribution: Figure 2h presents the distribution of prediction errors across the dataset, with the distribution centered around zero and relatively symmetrical. This symmetry suggests that the model does not have a systematic bias in its predictions. However, the presence of some outliers indicates that while the model is generally accurate, there are instances where predictions deviate significantly from the actual values.

![image/png](https://cdn-uploads.huggingface.co/production/uploads/66c5cec3f720e602d299597a/3Iv0YjHOtkP9751_wc_Hm.png)

Summary of Results
The MethSurvPredictor model, built on an MLP architecture, demonstrates strong predictive capabilities for LGG patient prognosis using methylation data. The results, including the decrease in training and test losses, stabilization of the R² score, and the strong correlation between predicted and actual OS times, confirm that this model is an effective tool for clinical decision-making. Additionally, the detailed analysis of weight distributions and heatmaps provides further insight into how the model processes and prioritizes input features, leading to high-precision predictions.

References
Weinstein, J. N., et al. (2013). The Cancer Genome Atlas Pan-Cancer analysis project. Nature Genetics, 45(10), 1113-1120.
Goldman, M., et al. (2020). Visualizing and interpreting cancer genomics data via the Xena platform. Nature Biotechnology, 38(6), 675-678.
Paszke, A., et al. (2019). PyTorch: An Imperative Style, High-Performance Deep Learning Library. In Advances in Neural Information Processing Systems (pp. 8024-8035).
Kaplan, E. L., & Meier, P. (1958). Nonparametric estimation from incomplete observations. Journal of the American Statistical Association, 53(282), 457-481.
Jolliffe, I. T. (2002). Principal Component Analysis (2nd ed.). Springer.
Hinton, G. E., et al. (2012). Improving neural networks by preventing co-adaptation of feature detectors. arXiv preprint arXiv:1207.0580.
