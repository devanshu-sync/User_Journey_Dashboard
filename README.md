# 📊 User Journey Dashboard

A **Streamlit dashboard** to analyze **Amazon user search journeys** (klog typing, item, category searches) and **cart additions**.  
It connects to a ClickHouse API and provides insights like **conversion rates, drop-offs, and user activity**.  

---

## 🚀 Features

- Consolidates **KLOG typing sequences** into final product searches  
- Detects **products added to cart** within a user’s session  
- Interactive filters:
  - User tokens  
  - Search type (klog, category, item)  
  - Cart status  
  - Date range  
- 📈 Visual analytics:
  - Top products searched  
  - Conversion rate by search type  
  - Search activity over time  
  - User-level summaries  

---
## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run Ecom.py

