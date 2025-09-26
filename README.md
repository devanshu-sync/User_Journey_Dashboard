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
## 🧠 Core Klog Logic

## What is Klog Data?
- Klog (Keystroke Logging) data captures individual keystrokes as users type in search boxes
- creating sequences like: "i" → "ip" → "iph" → "ipho" → "iphone"
## Key Processing Steps

 1. Klog Sequence Consolidation
     Logic Flow:
     - Time Window Grouping: Groups klog entries from the same user within 5 minutes
     - Sequence Building: Builds typing sequences by following consecutive keystrokes
     - Completion Detection: Identifies completion via category/item events or longest klog entry
     - Product Name Extraction: Extracts final product name from the sequence

## 🛒 Cart Detection Logic

 Time-Based Correlation:
 - 24-hour window: Products searched within 24 hours of cart addition
 - Recency priority: Most recent search selected
 - User-specific matching: Matches searches and carts per user

## Edge Case Handling:
 - "All Categories": Finds actual product search before navigation
 - Duplicate Cart Events: Deduplication with set
 - Incomplete Sequences: Uses longest typed text
 🔧 Technical Implementation

## Performance Optimizations:
- Binary Search for time-based lookups
 - Hash Maps for fast user grouping
 - Caching: @st.cache_data(ttl=300)
 - Indexed Lookups for user events

     

---
## Installation

1. Install required packages:
```bash
pip install -r requirements.txt
```

2. Run the dashboard:
```bash
streamlit run Ecom.py

