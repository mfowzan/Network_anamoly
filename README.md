Network Anomaly Detection System

A machine learning–based Network Anomaly Detection System built using multiple models and an ensemble framework to detect malicious or abnormal network traffic. The system is trained on the NSL-KDD dataset and exposes a FastAPI backend for real-time and batch predictions.


---
Project Overview

With the rapid growth of network-based applications, detecting intrusions and abnormal behavior has become critical. This project implements multiple anomaly detection and classification models and combines them using an ensemble approach to improve accuracy and robustness.

The system can:

Detect network anomalies

Handle both known and unknown attacks

Provide model-wise and ensemble predictions

Serve predictions via REST APIs



---

Features

Multiple ML models (Unsupervised + Supervised)

Ensemble decision framework

Trained on industry-standard NSL-KDD dataset

FastAPI-based backend

Scalable and modular architecture



---

Machine Learning Models Used

1. Autoencoder (Deep Learning – Unsupervised)

Learns normal traffic patterns

High reconstruction error indicates anomaly

Effective for unknown and zero-day attacks


2. Isolation Forest (Unsupervised)

Isolates anomalies faster due to their rarity

Efficient and scalable


3. One-Class SVM (Unsupervised)

Learns boundary around normal data

Flags outliers as anomalies


4. Random Forest Classifier (Supervised)

Ensemble of decision trees

High accuracy and interpretability


5. XGBoost Classifier (Supervised)

Gradient boosting-based classifier

High performance and efficiency



---

 Ensemble Framework

Instead of relying on a single model, this system uses an ensemble strategy:

Each model predicts independently

Final decision is based on majority voting

Reduces false positives

Improves detection reliability


Why Ensemble?

> Different models capture different patterns. Combining them improves robustness and accuracy.




---

Dataset Used

NSL-KDD Dataset

Files: KDDTrain+.txt, KDDTest+.txt

Preprocessed and cleaned

Binary classification:

0 → Normal

1 → Attack




---

 Data Preprocessing

Removal of redundant features

Label encoding and cleaning

Feature scaling using StandardScaler

Feature consistency maintained across all models


---


Results & Performance

High detection accuracy using XGBoost

Autoencoder effective for unknown anomalies

Ensemble significantly reduces false positives



---
 Applications

Intrusion Detection Systems (IDS)

Network Monitoring

Cybersecurity Analytics

Real-time anomaly detection



---

Conclusion

This project demonstrates how combining multiple machine learning models using an ensemble framework improves network anomaly detection. The system is scalable, reliable, and suitable for real-world cybersecurity applications.


---

Author
Mohammad Fowzan
Computer Science Engineering Student
Focus: Machine Learning, Cybersecurity, AI Systems


---

License

This project is for academic and educational purposes.
