Hybrid Recommendation System
Balancing Personalization, Time-Aware Popularity, and Catalogue Exposure
A hybrid AI recommendation framework that combines personalized recommendation using SVD++ with time-aware popularity modeling using DecayPop to explore the trade-off between recommendation relevance, popularity bias, and broader catalogue exposure.
> **AI for Business Focus:** Designing recommendation strategies that can balance individual preferences with changing trends and business objectives.
---
Executive Summary
Recommendation systems help digital businesses connect users with relevant products, content, and services. However, optimizing only for personalization can reinforce popularity bias and concentrate recommendations around a limited set of items.
This project proposes a hybrid recommendation framework that combines two complementary signals:
SVD++ for personalized user preference modeling.
DecayPop for time-aware popularity modeling, where recent interactions receive greater influence.
The framework evaluates nine hybrid configurations across 10-fold cross-validation and investigates whether different user segments respond differently to changes in the personalization-popularity balance.
The goal is not simply to identify a single "best" recommendation ratio, but to understand the trade-offs between relevance, popularity, catalogue exposure, and user behaviour that can inform business-level recommendation strategies.
---
Key Highlights
Dimension	Project Scope
Recommendation models	SVD++ + DecayPop
Hybrid configurations	9
Cross-validation	10-fold
User segments	3
Accuracy metrics	Precision, Recall, F1, NDCG@10
Popularity metrics	ARP, Gini Index
Catalogue metric	Catalogue Coverage
Statistical analysis	Friedman test + post-hoc analysis
Hyperparameter optimization	Optuna
---
1. Business Problem
Recommendation systems are increasingly used by digital platforms to help users discover relevant products, movies, content, and services.
However, recommendation systems face an important business trade-off.
Highly personalized recommendations can effectively match individual preferences, but may repeatedly recommend a relatively small group of popular items. Conversely, popularity-based recommendations can capture current trends and broaden content exposure, but may provide less individualized recommendations.
This creates a practical question:
> **How can an AI recommendation system balance individual user preferences with changing popularity trends while maintaining broader catalogue exposure?**
This project investigates a hybrid recommendation strategy that combines personalization and time-aware popularity, while also examining whether the appropriate balance differs across user segments.
---
2. Proposed AI Solution
The proposed framework combines two complementary recommendation signals.
SVD++ — Personalized Recommendation
SVD++ serves as the personalized recommendation component. It learns latent representations from historical user-item interactions and captures individual user preferences.
DecayPop — Time-Aware Popularity
DecayPop introduces a time-aware popularity signal. More recent interactions receive greater influence than older interactions, allowing the popularity component to respond to changing trends.
Hybrid Recommendation
The two signals are combined using different weights.
The experiment evaluates nine configurations ranging from:
10% SVD++ : 90% DecayPop
to
90% SVD++ : 10% DecayPop
This creates an experimental framework for studying how recommendation outcomes change as the system moves from trend-oriented recommendation toward personalized recommendation.
---
3. Why This Hybrid Approach?
The two components provide complementary capabilities.
Component	Strength	Limitation
SVD++	Captures individual user preferences	May concentrate recommendations around repeatedly preferred items
DecayPop	Captures recent popularity trends	Provides less individualized recommendations
Hybrid Model	Combines personalization and time-aware popularity	Requires an appropriate balance between the two signals
Rather than treating personalization and popularity as competing approaches, this project treats them as controllable signals within a recommendation strategy.
This enables businesses to investigate different operating points depending on their objectives.
---
4. Business Value Proposition
The central business value of the framework is flexibility.
Different businesses may prioritize different outcomes:
Business Objective	Potential Recommendation Priority
Maximize relevance	Stronger personalization
Promote emerging trends	Stronger time-aware popularity
Increase catalogue exposure	Greater popularity/exploration contribution
Serve different user behaviours	Segment-aware weighting
Balance multiple objectives	Hybrid recommendation
Therefore, the hybrid ratio can be viewed as a business decision parameter, rather than merely a model hyperparameter.
This creates a path toward recommendation policies that can be adjusted according to business objectives, user behaviour, and catalogue strategy.
---
5. Research Questions
This project investigates two main questions:
RQ1. How do different hybrid ratios of personalized and popularity-based recommendation affect recommendation accuracy and catalogue coverage?
RQ2. Does sensitivity to changes in hybrid ratios differ across user segments, and which segments exhibit the greatest sensitivity?
---
6. Dataset
The experiment uses the MovieLens 100K dataset.
Property	Value
Original ratings	100,000
Users	943
Movies	1,682
Minimum user interactions	5
Minimum item interactions	10
Ratings after filtering	97,953
Users with fewer than five interactions and items with fewer than ten interactions are excluded during preprocessing.
The raw and processed datasets are not included in this repository. This keeps the repository lightweight while allowing the experimental implementation and selected results to remain available for reproducibility.
---
7. Experimental Methodology
The experimental workflow consists of:
Data preprocessing and filtering
10-fold cross-validation
SVD++ training and evaluation
Time-aware popularity modeling using DecayPop
Hyperparameter optimization using Optuna
Hybrid score construction
Evaluation across nine hybrid configurations
User segmentation
Accuracy, popularity, and catalogue-level evaluation
Statistical sensitivity analysis
Overall Pipeline
```text
                    User-Item Interactions
                              │
                              ▼
                    Data Preprocessing
                              │
                              ▼
                     10-Fold Cross-Validation
                              │
                  ┌───────────┴───────────┐
                  ▼                       ▼
                SVD++                  DecayPop
          Personalized Signal      Time-Aware Popularity
                  │                       │
                  └───────────┬───────────┘
                              ▼
                    Hybrid Recommendation
                              │
                              ▼
                  9 Hybrid Configurations
                              │
                              ▼
                      User Segmentation
                              │
                              ▼
                  Evaluation & Sensitivity
                              │
                              ▼
                    Business-Oriented Insights
```
---
8. Hybrid Configurations
Nine hybrid configurations are evaluated by varying the contribution of SVD++ and DecayPop.
Configuration	SVD++	DecayPop
10:90	10%	90%
20:80	20%	80%
30:70	30%	70%
40:60	40%	60%
50:50	50%	50%
60:40	60%	40%
70:30	70%	30%
80:20	80%	20%
90:10	90%	10%
The purpose of evaluating multiple configurations is to characterize the trade-off curve, rather than assuming that a single weighting strategy is universally optimal.
---
9. User Segmentation
The project investigates whether different types of users respond differently to changes in the hybrid ratio.
Three user segments are evaluated:
New Users
Regular Users
Trend-Followers
Sensitivity analysis examines whether changes in the SVD++ and DecayPop contributions produce consistent effects across these segments.
This is particularly relevant from a business perspective because a single recommendation policy may not be equally appropriate for every type of user.
---
10. Evaluation Framework
The system is evaluated from several complementary perspectives.
Recommendation Accuracy
Precision
Recall
F1-score
NDCG@10
These metrics evaluate the relevance and ranking quality of recommendations.
Popularity and Concentration
Average Recommendation Popularity (ARP)
Gini Index
These metrics provide insight into popularity bias and recommendation concentration.
Catalogue Exposure
Catalogue Coverage
Catalogue coverage measures how broadly the recommendation system exposes items across the available catalogue.
Statistical Analysis
Sensitivity to hybrid-ratio changes is evaluated using:
Friedman test
Post-hoc analysis
The combination of these metrics allows the system to be evaluated beyond accuracy alone.
---
11. Key Results
The experiments demonstrate that changing the relative contribution of personalization and time-aware popularity produces different effects across recommendation objectives.
Selected results include:
Measure	Result
SVD++ RMSE	0.9108
SVD++ MAE	0.7058
DecayPop ARP	50.68%
SVD++ Catalogue Coverage	0.7266
Hybrid Catalogue Coverage	0.7265–0.7576
DecayPop Test-only Catalogue Coverage	0.7731
The results indicate a trade-off between recommendation accuracy and catalogue exposure as the relative contribution of SVD++ and DecayPop changes.
Importantly, the effect of the hybrid ratio is not uniform across users. Different user segments demonstrate different levels of sensitivity to changes in the personalization-popularity balance.
The resulting insight is therefore not simply:
> *Which ratio is best?*
but rather:
> **How should a recommendation strategy balance personalization and popularity for different objectives and user behaviours?**
This perspective is more directly applicable to business decision-making.
---
12. From Research Finding to Business Decision
The framework provides a conceptual path from model evaluation to recommendation policy.
```text
Experimental Results
        │
        ▼
Identify Trade-offs
        │
        ▼
Understand User Segments
        │
        ▼
Select Recommendation Strategy
        │
        ▼
Align with Business Objective
        │
        ▼
Potential Deployment & A/B Testing
```
For example, a business may choose a more personalization-oriented strategy when relevance is the dominant objective, while increasing the contribution of time-aware popularity when trend responsiveness or broader catalogue exposure is more important.
The appropriate strategy should ultimately be validated against real business KPIs.
---
13. Potential Business Applications
The framework can potentially support recommendation-driven businesses such as:
E-commerce — product discovery and catalogue exposure
Streaming platforms — movie, music, or content recommendation
Digital content platforms — personalized content discovery
Online marketplaces — balancing personalized and trending products
Digital services — personalized discovery and user engagement
Potential business benefits include:
improved personalization,
better utilization of recent trends,
broader catalogue exposure,
reduced dependence on a small group of highly popular items,
segment-aware recommendation strategies, and
flexible recommendation policies aligned with business objectives.
---
14. Repository Structure
```text
.
├── notebooks/
│   └── 01_sanity_check.ipynb
│
├── src/
│   ├── config.py
│   ├── decaypop.py
│   ├── hybrid.py
│   ├── metrics.py
│   ├── preprocessing.py
│   ├── segmentation.py
│   ├── sensitivity.py
│   └── svdpp.py
│
├── scripts/
│   ├── 1. build_folds.py
│   ├── 1a. tune_svdpp.py
│   ├── 2a. train_svdpp.py
│   ├── 2b. train_decaypop.py
│   ├── 3. generate_hybrid_predictions.py
│   ├── 4. compute_metrics.py
│   ├── 5. build_segments.py
│   └── 6. user_sensitivity.py
│
├── tests/
│   └── test_parity.py
│
├── results/
│   ├── decaypop/
│   ├── metrics/
│   ├── segmentation/
│   ├── sensitivity/
│   ├── svdpp/
│   └── svdpp_tuning/
│
├── requirements.txt
├── index.py
└── README.md
```
Large raw datasets, processed datasets, model files, and selected intermediate outputs are excluded from version control.
---
15. Reproducibility
The repository provides the implementation, experiment configuration, selected results, and evaluation components used in this study.
To reproduce the experiment:
Clone the repository.
Install the required dependencies.
Obtain the MovieLens 100K dataset.
Place the dataset according to the expected project structure.
Run the preprocessing pipeline.
Execute the model training and evaluation scripts.
Generate the evaluation results.
The project uses 10-fold cross-validation to evaluate the recommendation framework across multiple train-test partitions.
Raw and intermediate datasets are intentionally excluded from version control because of file size and distribution considerations.
---
16. Installation
Clone the repository:
```bash
git clone https://github.com/riasiti-j/hybrid-svdpp-decaypop.git
cd hybrid-svdpp-decaypop
```
Install the required Python packages:
```bash
pip install -r requirements.txt
```
---
17. Technologies
The project is implemented primarily in Python using:
Python
Pandas
NumPy
Scikit-learn
Surprise
Optuna
Jupyter Notebook
Git / GitHub
The SVD++ component is implemented using the `SVDpp` implementation provided by the Surprise recommendation library.
---
18. Future Development
The current study provides a research prototype and benchmark evaluation. Several directions can extend it toward real-world business validation:
1. Domain-Specific Validation
Evaluate the framework using larger and domain-specific datasets such as e-commerce, streaming, or marketplace data.
2. Dynamic User-Aware Weighting
Develop dynamic hybrid weights based on user characteristics and behavioural signals rather than using a fixed ratio for all users.
3. Real-Time Recommendation
Extend the time-aware popularity component toward continuously updated recommendation environments.
4. Online Experimentation
Validate recommendation strategies using A/B testing or controlled online experiments.
5. Business KPI Evaluation
Move beyond offline recommendation metrics toward business-level outcomes such as:
engagement,
conversion,
retention,
click-through rate,
catalogue exposure, and
revenue.
This progression represents the pathway from an offline research prototype toward a deployable AI recommendation solution.
---
19. Limitations
This project is currently evaluated using the MovieLens benchmark dataset and offline recommendation metrics.
Therefore, the results should not be interpreted as direct evidence of commercial impact.
In particular:
the dataset does not represent a specific commercial domain,
business KPIs such as conversion and revenue are not directly measured,
online user behaviour has not yet been evaluated, and
deployment-level latency and scalability have not yet been tested.
These limitations define the next stage of development toward real-world business validation.
---
20. Project Context
This project explores how AI-based recommendation strategies can balance:
Personalization + Time-Aware Trends + Catalogue Exposure
The broader objective is to investigate how recommendation algorithms can move beyond optimizing a single accuracy metric and instead support multi-objective business decision-making.
The framework connects:
```text
AI / Machine Learning
        │
        ▼
Recommendation System
        │
        ▼
User Behaviour
        │
        ▼
Business Objectives
        │
        ▼
Recommendation Policy
```
---
License
This repository is intended for research and educational purposes.
