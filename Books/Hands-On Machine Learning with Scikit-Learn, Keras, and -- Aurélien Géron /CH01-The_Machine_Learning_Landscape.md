# Hands-On Machine Learning — Study Notes

**Book:** Hands-On Machine Learning with Scikit-Learn, Keras & TensorFlow  
**Author:** Aurélien Géron

## 📌 December 26, 2025
## Chapter 1: The Machine Learning Landscape

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

## 1️⃣ Supervised Learning
**Definition:**  
The model is trained using **labeled data**.

### 📧 Classification Example — Spam Detection
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

### 🚗 Regression Example — Car Price Prediction
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

### 🎯 Logistic Regression

- Outputs probability (0–1)  
- Example: `P(spam | X) = 0.2`  
- Threshold:
  - ≥ 0.5 → spam  
  - < 0.5 → not spam  

### ⭐ Common Supervised Learning Algorithms

- k-Nearest Neighbors  
- Linear Regression  
- Logistic Regression  
- Support Vector Machines (SVMs) 
- Decision Trees and Random Forests  
- Neural Networks  


## 2️⃣ Unsupervised Learning

**Definition:**  
The model finds patterns in **unlabeled data**.

### 📧 Example — Emails Without Labels
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

### ⭐ Common Unsupervised Learning Algorithms
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

### 1️⃣ Clustering
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

### 2️⃣ Visualization
**Visualization algorithms help us see complex data.**
- **Input**: high-dimensional, unlabeled data
- **Output**: a 2D or 3D plot

**Goal:**
- Keep similar data points close together
- Keep different clusters separated

This helps humans understand the structure of the data and spot unexpected patterns.

### 3️⃣ t-SNE (Example of Visualization)
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

### 4️⃣ Dimensionality Reduction
Goal:
- To reduce the number of features.
- Without losing too much information.

Example:

<img width="700" height="2999" alt="image" src="https://github.com/user-attachments/assets/7d043e3c-e0bc-49e6-89aa-1e33fbd97bc1" />

- Car Age + Mileage → combined into 1 feature.
- New Feature = Car wear and tear.
This is called feature extraction.
Result: The data becomes simpler, and the model becomes more efficient.

### 5️⃣ Anomaly Detection
Anomaly detection finds unusual or abnormal data points.

**Examples:**
- Credit card fraud
- Manufacturing defects
- Outliers in datasets

**How it works:**
- The model is trained using normal data.
- If new data looks very different → labeled as an anomaly

### 6️⃣ Anomaly Detection vs Novelty Detection

<img width="700" height="460" alt="image" src="https://github.com/user-attachments/assets/4574f188-272e-470d-bff7-1c3e8675b2b6" />

**🔴 Anomaly Detection**
- Training data: May be "dirty" (contain noise or outliers).
- Very rare patterns -> considered as anomaly
- Example: A Chihuahua (representing 1% of dogs) → might be flagged as an anomaly (because it looks different from the majority).

**🟢 Novelty Detection**
- Training data must be very clean
- The model only recognizes what is "normal."
- Example: A Chihuahua is not considered strange, because it is still recognized as a dog (assuming "dog" is the normal class).
<p>➡️ The Difference: It lies in the cleanliness of the training data and the specific goal of the detection.</p>

### 7️⃣ Association Rule Learning
Association rule learning finds relationships between items in large datasets.

**Goal:**
- To discover relationships between items.
- This is not about prediction, but rather identifying habitual patterns.

**Example:**
- Buy BBQ Sauce + Chips → Often results in buying Steak.
- Business Solution: Place these items near each other in the store.
<p>➡️ Application: Widely used in retail and product recommendation systems.</p>

### 3️⃣ Semi-Supervised Learning
**What it is**
Semi-supervised learning uses:
   - A small amount of labeled data
   - A large amount of unlabeled data
<p>This is common because labeling data is expensive and slow.</p>

<img width="700" height="496" alt="image" src="https://github.com/user-attachments/assets/7a83bdc0-012b-43ec-9922-8704e975308f" />

**What the picture means**
   - Triangles & squares → labeled data (we know their class)
   - Gray dots (circles) → unlabeled data
   - Cross (X) → a new data point we want to classify

Even if the cross is closer to labeled squares, the unlabeled data structure shows it belongs to the triangle cluster.
Unlabeled data helps guide the decision.

**Example: (Google Photos)**
1. You upload many photos
   You upload hundreds of family photos.
   - No names
   - No labels
   - Just raw photos
     
2. The system groups faces (UNSUPERVISED)
   The system looks at faces and says:
   - “These faces look the same” → Group A
   - “These faces look the same” → Group B

   Example:
      - Person A appears in photos 1, 5, 11
      - Person B appears in photos 2, 5, 7

   At this point:
      - The system does NOT know names
      - It only knows which faces are similar
   
   This is unsupervised learning (clustering).   

3. You give ONE label per person
   Now the system asks you:
   > — “Who is this person?”

   You answer:
      - “This is Mom”
      - “This is Dad”
   
   You label just one face per group.

4. The system labels EVERYTHING (SUPERVISED part)
   Because it already grouped faces:
      - All faces in Group A → labeled Mom
      - All faces in Group B → labeled Dad

   Now:
      - Every photo is labeled
      - You can search: “Mom”
      - Google Photos shows all photos of Mom
  
### 4️⃣ Reinforcement Learning
Definition
Reinforcement Learning (RL) is very different from supervised learning.

There are:
- Agent → the learner
- Environment → the world
- Actions → what the agent can do
- Rewards / penalties → feedback

The goal:
  Learn the best strategy (policy) to maximize total reward.

Example Explanation:
<img width="700" height="823" alt="image" src="https://github.com/user-attachments/assets/cbc10e3a-8765-454a-9321-76b11bbde0fc" />

<p>🔥 Touch fire → negative reward</p>
<p>💧 Move to water → positive outcome</p>

So next time, the agent avoids fire.

**Example: (A Robot Learns to Walk)**
1. Define the Environment
   The environment is the maze itself:
      - Maze layout (walls, paths)
      - Start position
      - Exit position
      - Rules of movement
   The environment controls what is allowed and what happens next.

2. Define the Agent
   The agent is the game character.
      - It does not know the maze
      - It must learn by trying
  
3. Define the State (Observation)
  The state is what the agent can “see”, for example:
      - Its current position: (x, y)
      - Nearby walls (up, down, left, right)
      - Whether it is at the goal
   <p>This is the information the agent uses to decide.</p>

4. Define the Actions
   The actions are simple:
      - Move up
      - Move down
      - Move left
      - Move right
   <p>Each step, the agent chooses one action.</p>p>

5. Define the Reward (Goal)
   Rewards tell the agent what is good or bad:
      - +100 → reaches the exit
      - −1 → each step (to encourage faster solutions)
      - −10 → hits a wall
      - −100 → falls into a trap (if any)
   <p>Reward = feedback.</p>

6. Start an Episode (One Training Run)
   One episode = one full game:
      - Start at beginning
      - End when:
           - the agent reaches the exit, or
           - time limit is reached

7. Loop: Observe → Act → Get Reward
   Repeated many times per episode:
      a) Agent observes state (current position)
      b) Agent chooses an action (move)
      c) Environment executes the action
      d) Agent receives reward
      e) New state is observed
    <p>This happens many times in one episode.</p>
    
8. Update the Policy (Learning Step)
   After many steps or episodes:
      - The agent updates its policy
      - Actions that lead closer to the exit get higher probability
      - Actions that hit walls or traps are avoided
   <p>This is where learning actually happens.</p>

9. Repeat Many Times
   - First episodes: random movement, lots of mistakes
   - Later episodes: smarter paths
   - Eventually: the shortest path to the exit
    
10. Deployment (Learning OFF)
    When the game is released:
       - Learning is turned off
       - The agent only follows the learned policy
       - Behavior is stable and predictable
    <p>(Exactly like AlphaGo during real matches.)</p>

## Batch Learning vs Online Learning
### 🅰️ Batch Learning (Offline Learning)
**What it means**
   - The model is trained using all available data at once
   - It cannot learn new data by itself
   - Training is done offline
   - After training, the model is fixed
     
**If new data appears:**
   - You must retrain the whole model from scratch
   - Using old data + new data
   - Then replace the old model

**Why it’s called offline learning**
Because:
   - Training takes a lot of time
   - Uses a lot of CPU, memory, disk
   - So it’s done not while the system is running

**Problems with Batch Learning**
   - Training can take hours or days
   - Expensive if data is large
   - Not good for rapidly changing data (e.g. stock prices)
   - Not suitable for devices with limited resources (phones, robots)

**Simple example**
📧 Spam filter (batch learning)
   - Train model today using all emails
   - Tomorrow a new spam type appears
   - ❌ Model doesn’t know it
   - ✅ Must retrain the whole model again

### 🅱️ Online Learning (Incremental Learning)
**What it means**
   - The model learns step by step
   - New data is added one by one or in small batches (mini-batches)
   - The model keeps updating itself

**This happens:**
   - While the system is running
   - As new data arrives

<img width="700" height="506" alt="image" src="https://github.com/user-attachments/assets/3a655cd9-13bb-47fb-adf1-127284cf3cbf" />

**What the Picture above means** 
1. Train the model
2. Launch it into production
3. New data comes in
4. The model:
      - Runs
      - Learns
5. Updates itself continuously
<p>The model never stops learning</p>

**Advantages of Online Learning**
   - Very fast updates
   - Uses less memory
   - Good for:
        - Streaming data (stock prices, sensors)
        - Limited hardware
        - Real-time systems

   - After learning:
        - Old data can be discarded
        - Saves storage space
