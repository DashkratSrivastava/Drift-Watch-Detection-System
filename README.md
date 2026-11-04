# DriftWatch – Data Drift Detection System

## Overview
**DriftWatch** is a data monitoring system designed to detect **data drift** in machine learning pipelines. It compares incoming data with training data to identify changes in statistical distributions that may affect model performance.

Instead of focusing only on model accuracy, DriftWatch emphasizes **post-deployment monitoring**, which is critical in real-world AI systems.

---

## Objective
The main goal of this project is to:

- Detect distribution changes in incoming data  
- Quantify drift using statistical metrics  
- Identify the most affected features  
- Provide visual insights through an interactive dashboard  

---

## ⚙️ Features

| Feature                     | Description                                                                 |
|---------------------------|-----------------------------------------------------------------------------|
| Data Upload               | Upload CSV datasets for analysis                                            |
| Automatic Data Splitting  | Splits data into training and simulated incoming data                       |
| Drift Simulation          | Introduces controlled drift for demonstration                               |
| Feature-wise Analysis     | Calculates drift for each numerical feature                                 |
| Drift Metrics             | Uses Mean Difference and PSI (Population Stability Index)                  |
| Drift Classification      | Classifies drift as Low, Medium, or High                                    |
| Top Feature Detection     | Identifies the most drifting feature                                        |
| Visualization             | Displays distribution comparison graphs                                     |

---

## Key Concepts

### Data Drift
Data drift occurs when the statistical properties of incoming data differ from the training data, leading to degraded model performance.

### Metrics Used

| Metric                  | Purpose                                                                 |
|------------------------|-------------------------------------------------------------------------|
| Mean Difference        | Measures shift in average values                                        |
| PSI (Population Stability Index) | Measures change in distribution between datasets              |

---

## System Architecture

```text
User Input (CSV)
↓
Data Ingestion & Preprocessing
↓
Drift Simulation
↓
Feature-wise Drift Calculation
↓
Drift Aggregation Engine
↓
Visualization Dashboard
```

---

## 🖥️ User Workflow

1. Upload a dataset (CSV file)  
2. System splits data into:
   - Training data (baseline)  
   - Incoming data (simulated)  
3. Drift is introduced in incoming data  
4. System calculates:
   - Feature-wise drift  
   - Overall drift score  
5. Results are displayed:
   - Drift status (Low / Medium / High)  
   - Top drifting feature  
   - Graphical comparison  

---

## Tech Stack

| Component     | Technology Used         |
|---------------|------------------------|
| Backend       | Python                 |
| Data Handling | Pandas, NumPy          |
| Visualization | Matplotlib             |
| Frontend      | Streamlit              |

---

## Project Structure

```text
DriftWatch/
│
├── app.py # Streamlit frontend
├── backend.py # Drift detection logic
├── dataset.csv # Sample dataset
├── README.md # Project documentation
```

---

## 🚀 How to Run

### 1. Install dependencies
```bash
pip install streamlit pandas numpy matplotlib
```
### 2. Run the application
```bash
streamlit run app.py
```

## Future Enhancements
Real-time data monitoring
Integration with ML pipelines
Support for categorical drift (Chi-square test)
Automated model retraining triggers

## Conclusion
DriftWatch demonstrates the importance of monitoring data after deployment. By detecting early signs of drift, it helps maintain model reliability and supports better decision-making in production systems.

## 🤝 Contributors
- **Dashkrat Srivastava** ([@DashkratSrivastava](https://github.com/DashkratSrivastava))
