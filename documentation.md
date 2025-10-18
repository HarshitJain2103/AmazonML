# 🧠 ML Challenge 2025: Smart Product Pricing Solution

**🏷️ Team Name:** Cosmic  
**👥 Team Members:** Kirty Gupta, Sarthak Jha, Harshit Jain, Adithya  
**📅 Submission Date:** October 13, 2025  

---

## 🧾 1. Executive Summary

Our solution employs a **multimodal ensemble learning approach** for product price prediction, integrating **catalog data, text embeddings, and image embeddings**.

We combined three base learners — **XGBoost**, **LightGBM**, and **CatBoost** — followed by a **SMAPE-optimized meta-blending ensemble**.

After optimization, our model achieved a **Final Out-of-Fold (OOF) SMAPE of 46.34%**, demonstrating strong generalization and stability across folds.

---

## 🔍 2. Methodology Overview

### 🧩 2.1 Problem Analysis
The challenge involves predicting product prices using both **structured (catalog)** and **unstructured (text and image embeddings)** data.

Key insights from data exploration:
- Text and image embeddings capture **complementary semantic information**.  
- **Normalization of units** (e.g., kg, g, ml, L) and quantities improved feature consistency.  
- **Brand and product categories** exhibited clear and meaningful price clusters.  

---

### 🏗️ 2.2 Solution Strategy

We designed a **two-level ensemble pipeline** for robust and accurate prediction.

#### **Base Models**
- Gradient-boosted regressors: **XGBoost**, **LightGBM**, **CatBoost**  
- Trained using **7-fold cross-validation** on combined feature sets (text, image, catalog).

#### **Meta-Ensemble**
- Used **RidgeCV-based SMAPE-optimized blending** on OOF predictions.  
- Automatically tuned ensemble weights to minimize SMAPE.  

**Final Ensemble Weights:**
- 🟦 XGBoost → 0.770  
- 🟩 LightGBM → 0.230  
- 🟨 CatBoost → 0.000  

---

## 🧱 3. Model Architecture

### ⚙️ 3.1 Architecture Overview

```mermaid
flowchart TD
    A[Catalog Data] --> B[Text Embeddings (CLIP/Nomic)]
    A --> C[Image Embeddings]
    B --> D[PCA Reduction (256D)]
    C --> E[PCA Reduction (128D)]
    D --> F[Cross-Modal Feature Engineering]
    E --> F
    F --> G[Structured Catalog Parsing + Feature Join]
    G --> H[XGBoost Model]
    G --> I[LightGBM Model]
    G --> J[CatBoost Model]
    H --> K[OOF Predictions]
    I --> K
    J --> K
    K --> L[RidgeCV Meta-Ensemble (SMAPE Optimized)]
    L --> M[Final Price Predictions]
```

---

## 📊 4. Model Performance

### 🧮 4.1 Validation Results

| Model | OOF SMAPE (%) |
|:------|---------------:|
| **XGBoost** | 46.39 |
| **LightGBM** | 46.93 |
| **CatBoost** | 47.82 |
| 🏁 **Final Ensemble (Optimized)** | **46.34** |

**Blend Weights:**  
- XGB = 0.770  
- LGB = 0.230  
- CAT = 0.000  

**Predicted Price Range:** $1.46 – $174.08  
**Mean Predicted Price:** $18.71  
**Median Predicted Price:** $13.82  

---

### ⚡ 4.2 Improvements

| Enhancement | Description | Impact |
|:-------------|:------------|:--------|
| **Higher PCA Dimensions** | Text 256D, Image 128D | Better semantic representation |
| **Cross-Modal Features** | Added cosine & L2 distance similarities | Improved multimodal correlation |
| **Early Stopping + Tuned Params** | Stabilized boosting models | Reduced overfitting |
| **SMAPE-Optimized Blending** | Grid-searched ensemble weights | Improved overall accuracy |

---

## 🏁 5. Conclusion

Our final model effectively integrates **semantic (text)**, **visual (image)**, and **structured (catalog)** information through a hierarchical ensemble pipeline.

The system achieved a **Final OOF SMAPE of 46.34%**, outperforming all individual base models and providing consistent price estimation across diverse product types.

### 🔮 Future Work
- Fine-tune text/image embeddings using **contrastive multimodal transformers**.  
- Incorporate **brand-level priors** and **discount metadata**.  
- Add **uncertainty estimation** for confidence-aware price prediction.

---

## 📂 Appendix

### 🧰 A. Code Artefacts
**Full Code Directory:** [Google Drive Link — _https://drive.google.com/drive/folders/1L6Ams_PjVvR9SzEafiYdmg7CXQim9XWJ?usp=drive_link_]
**Drive Link where predictions are saved:** [Google Drive Link — _https://drive.google.com/drive/u/0/folders/1JFQpKQWBTF6fZV0LvCF5YyrGcijz0uW0_]

---

### 📈 B. Additional Results

| Setting | Value |
|:---------|:-------|
| **Cross-validation** | 7 folds, stratified by log(price) |
| **Metric** | Symmetric Mean Absolute Percentage Error (SMAPE) |
| **Scaler** | RobustScaler for feature normalization |
| **Price Clipping** | 0.5th–99.5th percentile for prediction stability |

---

> 🧩 **Summary:**  
> Through category-aware modeling, multimodal feature engineering, and SMAPE-optimized blending, the team successfully developed a high-performing price prediction system achieving **46.34% SMAPE**, ready for real-world deployment.

---
