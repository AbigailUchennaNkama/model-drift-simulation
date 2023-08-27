**Simulating Model Drift**

**Aim**
The primary aim of the "Simulating Model Drift" project is to demonstrate the phenomenon of model deterioration or drift in performance when exposed to unseen or slightly altered data. This project provides insights into how machine learning models, which perform well during their initial training, might degrade in accuracy and reliability over time due to changes in the data distribution.

#**Table of Contents**
* <ins>**Introduction**</ins>
* <ins>**What is Model Drift?**</ins>
  * <ins>**Project Scope**</ins>
###* <ins>**Usage**</ins>
###* <ins>**Implementation**</ins>


**Introduction**
Machine and Deep learning models are trained to make accurate predictions based on specific datasets. However, as time passes, the data distribution can change due to various factors such as shifts in user behavior, environmental changes, or evolving trends or in this case changes in the image. These changes in data distribution can lead to what is called "model drift."

Model drift occurs when a machine learning model's performance deteriorates on new or modified data that it hasn't encountered during its training phase. Detecting and addressing model drift is crucial to maintaining the reliability and effectiveness of machine learning systems in real-world applications.

**What is Model Drift?**
Model drift refers to the deterioration in the performance of a machine or Deep learning model over time due to changes in the input data. This deterioration can lead to incorrect predictions, reduced accuracy, and also a loss of trust in the model's output. Model drift can occur due to various reasons, including:

* **Data Evolution:** The underlying patterns and relationships in the data change over time, leading the model to make inaccurate predictions.

* **Concept Drift:** The concept being predicted changes, rendering the model's learned relationships obsolete.

* **Covariate Shift:** The distribution of the input data changes, making the model's learned features less relevant or accurate.

**Project Scope**
The "Simulating Model Drift" project aims to create a controlled environment for observing and studying the effects of changes in input data resulting in model drift. It provides a framework to:

[^1]: **Generate Data**: Create datasets with controlled variations to simulate changes in image data .

**[^1]:Train Initial Model:** finetune a pretrained Resnet18 model on the initial dataset to establish a baseline performance.

**[^1]: Introduce Drift:** Gradually introduce drift by modifying the data and introducing new data patterns (data augmentation).

**[^1]: Evaluate Drift:** Monitor the model's performance over time and analyze how its accuracy and reliability change as drift is introduced.

**[^1]: Visualize Results:** Visualize the impact of model drift through graphs, charts, and performance metrics.

**Implementation**
The project is implemented in Python, utilizing popular machine learning libraries such as pytorch and matplotlib. The codebase is organized into modules for data generation, model training, drift introduction(data augmentation), evaluation, and visualization.


