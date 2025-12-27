<h2>📘Notes of Hands-On Machine Learning with Scikit-Learn, Keras, & TF </h2>
Author: Aurélien Géron

<h4><b>📌 Date: Friday, December 26, 2025</b></h4>
<h4>📘 Chapter 1: The Machine Learning Landscape</h4>
<h3>What is Machine Learning? 🤔🤔🤔</h3>
<pre>
   Machine Learning is the] field of study that gives computers the ability to learn without being explicitly programmed.
   <i>~Arthur Samuel, 1959</i>
</pre>
<h3>First ML: 1990s: the spam filter 😮</h3>
<pre>
   Flag which email was a spam 
</pre>


<h3>
   📩 Key Concepts (Spam Filter Example):
</h3>

<ul>
   <li>Training Set: Given an example of spam emails (e.g. flagged by users) and example of regular (nonspam / "ham")</li>
   <li>Training instance / sample: Each training example</li>

   
   <li>Task (T): To flag for new emails</li>
   <li>Experience (E): Experience is the training data</li>
   <li>Performance (P): Performance measure needs to be defined</li>
      <li>Ex: Ratio of correctly classified emails (Accuracy)</li>
</ul>

<p>➡️ ML improves P on T using E.</p>

<h3>Why Use Machine Learning?</h3>
<p>Consider how you would write a spam filter using traditional programming techniques</p>
<ol>
   <li>Consider what spams looks like. Could be some words or phrases that typically appears the most in the subject (eg. "4U", "credit card", "free", and "amazing". We could also consider like other patterns in the
sender’s name, the email’s body, and other parts of the email.</li>
   <li>Write a detection algorithm for each of the patterns that you noticed, and your program would flag emails as spam if a number of these patterns were detected. </li>
   <li>Test your program and repeat steps 1 and 2 until it was good enough to launch</li>
</ol>
<p>Since the problem is difficult, your program will likely become a long list of complex rules—pretty hard to maintain.😩</p>


<h3>📝Examples of Applications</h3>
<ol>
   <li>Analyzing Images Of Products On A Production Line To Automatically Classify Them</li>
   <ul>
      <li>This is image classification, typically performed using convolutional neural networks
      <a href="https://www.w3schools.com">CH 14</a></li>
   </ul>
   
   <li>Detecting Tumors in Brain Scans </li>
   <ul>
      <li>This is semantic segmentation, where each pixel in the image is classified (as we want to determine the exact location and shape of tumors), typically using CNNs as well.</li>
   </ul>
   
   <li>Automatically Classifying News Articles</li>
   <ul>
      <li>This is natural language processing (NLP), and more specifically text classification, which can be tackled using recurrent neural networks (RNNs), CNNs, or Transformers <a href="https://www.w3schools.com">CH 16</a></li>
   </ul>

   <li>Automatically Flagging Offensive Comments On Discussion Forums</li>
   <ul>
      <li>This is also text classification, using the same NLP tools.<a href="https://www.w3schools.com">CH 16</a></li>
   </ul>

   <li>Summarizing Long Documents Automatically</li>
   <ul>
      <li>This is a branch of NLP called text summarization, again using the same tools. <a href="https://www.w3schools.com">CH 16</a></li>
   </ul>

   <li>Creating A Chatbot Or A Personal Assistant</li>
   <ul>
      <li>This involves many NLP components, including natural language understanding (NLU) and question-answering modules. </li>
   </ul>
   
   <li>Forecasting Your Company’s Revenue Next Year, Based On Many Performance Metrics</li>
   <ul>
      <li>This is a regression task (i.e., predicting values) that may be tackled using any regression model, such as a Linear Regression or Polynomial Regression model (see Chapter 4), a regression SVM (see Chapter 5), a regression Random Forest (see Chapter 7), or an artificial neural network (see Chapter 10). If you want to take into account sequences of past performance metrics, you may want to use RNNs, CNNs, or Transformers (see Chapters 15 and 16).  
      Making your app react to voice commands
This is speech recognition, which requires processing audio samples:
since they are long and complex sequences, they are typically
processed using RNNs, CNNs, or Transformers (see Chapters 15 and
16).</li>
   </ul>

   <li>Detecting Credit Card Fraud</li>
   <ul>
      <li>This is anomaly detection (see Chapter 9).</li>
   </ul>

   <li>Segmenting Clients Based On Their Purchases So That You Can Design A Different Marketing Strategy For Each Segment </li>
   <ul>
      <li>This is clustering (see Chapter 9).</li>
   </ul>

   <li>Representing A Complex, High-Dimensional Dataset In A Clear And Insightful Diagram </li>
   <ul>
      <li>This is data visualization, often involving dimensionality reduction techniques (see Chapter 8).</li>
   </ul>

   <li>Recommending A Product That A Client May Be Interested In, Based On Past Purchases</li>
   <ul>
      <li>This is a recommender system. One approach is to feed past purchases
(and other information about the client) to an artificial neural network
(see Chapter 10), and get it to output the most likely next purchase.
This neural net would typically be trained on past sequences of
purchases across all clients.
</li>
   </ul>

   <li>Building An Intelligent Bot For A Game/li>
   <ul>
      <li>This is often tackled using Reinforcement Learning (RL; see
Chapter 18), which is a branch of Machine Learning that trains agents
(such as bots) to pick the actions that will maximize their rewards over
time (e.g., a bot may get a reward every time the player loses some life
points), within a given environment (such as the game). The famous
AlphaGo program that beat the world champion at the game of Go was
built using RL.
</li>
   </ul>
   
</ol>

<hr>
<h4><b>📌 Date: Saturday, December 27, 2025</b></h4>
<h3>📝Types of Machine Learning Systems</h3>
<p>There are 4 major categories:</p>
<ol>
   <li>Supervised learning</li>
   <p>Supervised learning is a type of machine learning where the model is trained using labeled data — each training example has an input and the correct output (label).</p>
   <ol><h4>Example of Classification</h4></ol>
   <ol><h4>📧 Example 1: Spam classification</h4></ol>
   <ul>
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
      <p>X = (has_link, length, text features)</p>
      <p>y = spam / ham</p>
     <p>👉 Model learns: X → y</p>
      </ul>

   <ol><h4>Example of Predicting a Target Numeric Value</h4></ol>
   <ol><h4>🚗 Example 2: Regression (car price)</h4></ol>
   <ul>
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
      <p>X = mileage, age, brand</p>
      <p>y = price (number)</p>
     <p>👉 Model learns: X → y</p>
      <p>This is a Regression</p>
      <p>Regression is a type of supervised learning task where the goal is to predict a numeric (continuous) value, not a category.</p>
      </ul>
     <ol><h4>🔁 Algorithms vs Tasks</h4></ol>
     <ul>
        <p>Even though we usually say:</p>
        <ul>
           <li>Regression → predicts numbers</li>
           <li>Classification → predicts classes</li>
        </ul>
        <p>👉 Some algorithms are flexible and can be adapted to do either task.</p>
     </ul>
     <ol><h4>🎯 Example: Logistic Regression</h4></ol>
     <ol>Despite the name “regression”, it is mostly used for classification.</ol>
     <ol>What it actually outputs:
     <ul><li>a number between 0 and 1, e.g. 0.2</li></ul>
        We interpret it as:
        <ul><li>P(class = 1 | X) = 0.2 → 20% chance of being spam</li></ul>
        Then we apply a threshold:
        <ul><li>if ≥ 0.5 → predict spam</li></ul>
        <ul><li>else → not spam</li></ul>
        So:
        Logistic Regression does regression on probabilities, then we turn that into a class.
        That’s why it’s used for classification.
     </ol>
      <ol><h4>🎯Most Important Supervised Learning Algorithms</h4></ol>
      <ul>
        <ul>
           <li>k-Nearest Neighbors</li>
           <li>Linear Regression</li>
           <li>Logistic Regression</li>
           <li>Support Vector Machines (SVMs)</li>
           <li>Decision Trees and Random Forests</li>
           <li>Neural networks</li>
        </ul>
      </ul>
    <li>Unsupervised learning</li>
    <p>Training data is unlabeled</p>
    
</ol>
