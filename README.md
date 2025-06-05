# 🧠 Decision Tree Classifier - Manual Implementation

## 📌 Overview

This project implements a **Decision Tree Classifier** from scratch in three different programming languages:
- **Python** (required for all groups)
- **C++** (second implementation)
- **Rust** (third implementation)

The tree is built manually by calculating **entropy** and **information gain** for each attribute to determine the best splits. The project supports `.csv` and `.txt` datasets with configurable delimiters.

---

## 📚 Assignment Goals

- Understand how decision trees work in classification problems.
- Build a tree step-by-step with information gain calculations.
- Implement the algorithm without using external libraries.
- Visualize the resulting decision tree (text-based or optional image).
- Perform classification with user inputs after tree construction.

---

## 📁 Dataset Compatibility

- Format: `.csv` or `.txt`
- Delimiters: comma (`,`), dot (`.`), or others (configurable)
- Up to **10 columns** supported
- The **last column is always the target/output**

### Example Datasets:
- `weather.csv`
- `contact_lenses.csv`
- `breast_cancer.csv`

---

## ⚙️ Features

- Works with multiple datasets
- Step-by-step calculation of entropy and information gain
- Tree structure shown in console (text-based)
- Accepts user input for prediction
- Repeatable predictions with different inputs
- Clean and readable code, well-documented

---

## 🚫 Restrictions

- ❌ No external machine learning libraries (e.g. `sklearn`, `golearn`, `ai4r`, etc.)
- ✅ Only standard libraries allowed
- 🚫 No hardcoded datasets
- ✅ Must be able to work with any compatible dataset

---

## 🚀 How to Run

### 🐍 Python
```bash
python3 decision_tree.py <dataset_file.csv>

💻 C++
bash
Copy
Edit
g++ -std=c++20 -o decision_tree decision_tree.cpp
./decision_tree <dataset_file.csv>
🦀 Rust
bash
Copy
Edit
cargo run --release -- <dataset_file.csv>
🖥️ Example Output
text
Copy
Edit
[Step-by-step entropy and information gain logs]
[Text-based decision tree]
Enter input (comma-separated): sunny,hot,high,false
Prediction: no
