# Hands-On Machine Learning — Study Notes

**Book:** Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow  
**Author:** Aurélien Géron

## 📌 December 26, 2025
### Chapter 1: The Machine Learning Landscape

### What Is Machine Learning?

> Machine Learning is the field of study that gives computers the ability to learn without being explicitly programmed.  
> — Arthur Samuel (1959)

### First Popular ML Application (1990s): Spam Filter

**Goal:** Automatically decide whether an email is spam.

### Key Concepts

- Training set: labeled spam and non-spam emails (nonspam / ham)
- Training instance: one email
- Task (T): classify new emails
- Experience (E): training data
- Performance (P): accuracy

Machine learning improves performance (P) on task (T) using experience (E).

### Why Use Machine Learning?

Traditional spam filters rely on hand-written rules:
1. Identify spam patterns  
   - Keywords like *“free”*, *“credit card”*, *“amazing”*  
   - Sender name patterns, email body structure
2. Write rules for each pattern
3. Test and refine repeatedly

❌ Problems:
- Large rule sets  
- Hard to maintain  
- Easily broken by new spam tactics

✅ **Machine Learning adapts automatically by learning from data.**

## 📝 Examples of Machine Learning Applications

1. Image classification (CNNs — Ch. 14)  
2. Tumor detection in brain scans (semantic segmentation)  
3. News classification (NLP — Ch. 16)  
4. Offensive comment detection (NLP)  
5. Document summarization (NLP)  
6. Chatbots and virtual assistants  
7. Revenue forecasting (regression, time series)  
8. Credit card fraud detection (anomaly detection — Ch. 9)  
9. Customer segmentation (clustering — Ch. 9)  
10. Data visualization (dimensionality reduction — Ch. 8)  
11. Recommender systems  
12. Game-playing bots (reinforcement learning — Ch. 18)

<hr>

## 📌 December 27, 2025  
### Types of Machine Learning Systems

### 1️⃣ Supervised Learning
**Definition:**  
The model is trained using **labeled data**.

#### 📧 Classification Example — Spam Detection
<table>
      <tr>
        <th>Email text</th>
        <th>has_link</th>
        <th>length</th>
        <th>Label</th>
      </tr>
      <tr>
        <td>Win money now!!!</td>
        <td>1</td>
        <td>120</td>
        <td>spam</td>
      </tr>
      <tr>
        <td>Meeting at 3pm tomorrow</td>
        <td>0</td>
        <td>45</td>
        <td>ham</td>
      </tr>
      <tr>
        <td>Cheap pills, buy now</td>
        <td>1</td>
        <td>90</td>
        <td>spam</td>
      </tr>
</table>

- **X:** features (has_link, length, text)  
- **y:** label (spam / ham)

#### 🚗 Regression Example — Car Price Prediction
<table>
      <tr>
        <th>Mileage</th>
        <th>Age (years)</th>
        <th>Brand</th>
        <th>Price ($)</th>
      </tr>
      <tr>
        <td>50,000</td>
        <td>3</td>
        <td>Toyota</td>
        <td>15000</td>
      </tr>
      <tr>
        <td>80,000</td>
        <td>5</td>
        <td>Honda</td>
        <td>12000</td>
      </tr>
      <tr>
        <td>20,000</td>
        <td>1</td>
        <td>BMW</td>
        <td>28000</td>
      </tr>
</table>

- **X:** mileage, age, brand  
- **y:** price (numeric)

#### 🎯 Logistic Regression

- Outputs probability (0–1)  
- Example: `P(spam | X) = 0.2`  
- Threshold:
  - ≥ 0.5 → spam  
  - < 0.5 → not spam  

#### ⭐ Common Supervised Learning Algorithms

- k-Nearest Neighbors  
- Linear Regression  
- Logistic Regression  
- Support Vector Machines (SVMs) 
- Decision Trees and Random Forests  
- Neural Networks  


### 2️⃣ Unsupervised Learning

**Definition:**  
The model finds patterns in **unlabeled data**.

#### 📧 Example — Emails Without Labels
 <table>
   <tr>
     <th>Email text</th>
     <th>has_link</th>
     <th>length</th>
   </tr>
   <tr>
     <td>Win money now!!!</td>
     <td>1</td>
     <td>120</td>
   </tr>
   <tr>
     <td>Meeting at 3pm tomorrow</td>
     <td>0</td>
     <td>45</td>
   </tr>
   <tr>
     <td>Cheap pills, buy now</td>
     <td>1</td>
     <td>90</td>
   </tr>
 </table>

 Possible outcome:
- Group A → promotional  
- Group B → normal

#### ⭐ Common Unsupervised Learning Algorithms
**Clustering**
- K-Means
- DBSCAN
- Hierarchical Cluster Analysis (HCA)
  
**Anomaly detection and novelty detection**
- One-class SVM
- DBSCAN
- Isolation Forest
  
**Visualization and dimensionality reduction**
- Principal Component Analysis (PCA)
- Kernel PCA
- Locally Linear Embedding (LLE)
- t-Distributed Stochastic Neighbor Embedding (t-SNE)
  
**Association rule learning**
- Apriori
- Eclat

#### 🅰️ Clustering
**Clustering means grouping data points that are similar to each other, without labels.**
- You have unlabeled data (for example, blog visitor data).
- The algorithm searches for patterns on its own and groups similar data together.
- You do not tell it which is group A, B, or C.

**Example**

<img width="700" height="524" alt="image" src="https://github.com/user-attachments/assets/c91b2b7e-bfe2-41d1-ae44-261369c4e066" />

- 40% of visitors: Male, like comics, read at night.
- 20% of visitors: Young, like sci-fi, visit on weekends.
The algorithm discovers these patterns itself; you do not define them.

**Hierarchical clustering**
- First creates big groups
- Then splits them into smaller groups
- Useful for market segmentation or targeted content

#### 🅱️ Visualization
**Visualization algorithms help us see complex data.**
- Input: high-dimensional, unlabeled data
- Output: a 2D or 3D plot

**Goal:**
- Keep similar data points close together
- Keep different clusters separated

This helps humans understand the structure of the data and spot unexpected patterns.

#### 3️⃣ t-SNE (Example of Visualization)
**t-SNE is a visualization technique.**
- It converts high-dimensional data into 2D space
- Similar items appear close together

<img width="700" height="757" alt="image" src="https://github.com/user-attachments/assets/f424cacb-b374-4dc1-bf08-e01e3ac4821a" />


**In the image:**
- Each dot = an image (e.g., a cat, a dog, a car, etc.).
- t-SNE maps high-dimensional data → into 2D.
- Similar objects → cluster together.

**Meaning:**
- No labels during training
- The model still captures semantic similarities: It understands which items are "meaningfully" similar to each other.

#### 4️⃣ Dimensionality Reduction
Goal:
- To reduce the number of features.
- Without losing too much information.

Example:

<img width="700" height="2999" alt="image" src="https://github.com/user-attachments/assets/7d043e3c-e0bc-49e6-89aa-1e33fbd97bc1" />

- Car Age + Mileage → combined into 1 feature.
- New Feature = Car wear and tear.
This is called feature extraction.
➡️ Result: The data becomes simpler, and the model becomes more efficient.

#### 5️⃣ Anomaly Detection
Anomaly detection finds unusual or abnormal data points.

**Examples:**
- Credit card fraud
- Manufacturing defects
- Outliers in datasets

**How it works:**
- The model is trained using normal data.
- If new data looks very different → labeled as an anomaly

#### 6️⃣ Anomaly Detection vs Novelty Detection

<img width="700" height="460" alt="image" src="https://github.com/user-attachments/assets/4574f188-272e-470d-bff7-1c3e8675b2b6" />

**🔴 Anomaly Detection**
- Training data: May be "dirty" (contain noise or outliers).
- Very rare patterns -> considered as anomaly
- Example: A Chihuahua (representing 1% of dogs) → might be flagged as an anomaly (because it looks different from the majority).

**🟢 Novelty Detection**
- Training data must be very clean
- The model only recognizes what is "normal."
- Example: A Chihuahua is not considered strange, because it is still recognized as a dog (assuming "dog" is the normal class).
➡️ The Difference: It lies in the cleanliness of the training data and the specific goal of the detection.

#### 7️⃣ Association Rule Learning
Association rule learning finds relationships between items in large datasets.

**Goal:**
- To discover relationships between items.
- This is not about prediction, but rather identifying habitual patterns.

**Example:**
- Buy BBQ Sauce + Chips → Often results in buying Steak.
- Business Solution: Place these items near each other in the store.

➡️ Application: Widely used in retail and product recommendation systems.
