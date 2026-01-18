import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 1. Data Load karein (skiprows use kar rahe hain agar headers ka masla ho)
df = pd.read_csv('AAPL_data.csv', index_col=0)

# Masla Fix: 'Close' column ko number mein badlein aur jo garbage hai usay hata dein
df['Close'] = pd.to_numeric(df['Close'], errors='coerce')
df.dropna(subset=['Close'], inplace=True)

# 2. Feature Engineering
# fill_method=None deprecation warning ko khatam karne ke liye
df['Returns'] = df['Close'].pct_change(fill_method=None)
df['Volatility'] = df['Close'].rolling(window=21).std()

# Missing values (starting rows) ko drop karein
df.dropna(inplace=True)

# 3. Scaling
scaler = StandardScaler()
scaled_data = scaler.fit_transform(df[['Close', 'Returns', 'Volatility']])

# 4. Model 1: Isolation Forest
iso_forest = IsolationForest(contamination=0.02, random_state=42)
df['IF_Anomaly'] = iso_forest.fit_predict(scaled_data)

# 5. Model 2: One-Class SVM
oc_svm = OneClassSVM(nu=0.02, kernel="rbf", gamma='auto')
df['SVM_Anomaly'] = oc_svm.fit_predict(scaled_data)

# 6. Results
print(f"Total Trading Days: {len(df)}")
print("Isolation Forest Anomalies:", (df['IF_Anomaly'] == -1).sum())
print("One-Class SVM Anomalies:", (df['SVM_Anomaly'] == -1).sum())

# 7. Visualization
plt.figure(figsize=(12, 6))
plt.plot(df.index, df['Close'], label='AAPL Price', color='blue', alpha=0.4)

# Plot Isolation Forest Anomalies (Red)
if_anom = df[df['IF_Anomaly'] == -1]
plt.scatter(if_anom.index, if_anom['Close'], color='red', label='IF Anomaly', marker='x')

# Plot SVM Anomalies (Green circles)
svm_anom = df[df['SVM_Anomaly'] == -1]
plt.scatter(svm_anom.index, svm_anom['Close'], edgecolors='green', facecolors='none', s=80, label='SVM Anomaly')

plt.title('StockPulse: Comparison Model (IF vs SVM)')
plt.xticks(rotation=45)
plt.legend()
plt.tight_layout()
plt.show()