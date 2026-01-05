Credit Score Classification – Machine Learning Project

# Sommaire

- [Credit Score Classification – Machine Learning Project](#credit-score-classification--machine-learning-project)
- [Description du projet](#description-du-projet)
- [Données](#données)
- [Prétraitement & Feature Engineering](#pretraitement--feature-engineering)
- [Modèles testés](#modèles-testés)
- [Évaluation](#évaluation)
- [Conclusion](#conclusion)
- [Technologies utilisées](#technologies-utilisées)

# Description du projet

Ce projet vise à prédire le score de crédit des clients (Poor, Standard, Good) à partir de données financières et bancaires réelles issues de Kaggle.
Il s’agit d’un problème de classification supervisée multi-classes.

Le projet couvre l’ensemble du pipeline Machine Learning :

- Exploration des données

- Prétraitement avancé

- Feature engineering

- Entraînement et comparaison de plusieurs modèles

- Évaluation et analyse des performances

# Données

- 100 000 lignes

- 27 variables

- Données hétérogènes : numériques, catégorielles, binaires

- Données bruitées avec valeurs aberrantes et biais

# Prétraitement & Feature Engineering

- Nettoyage des valeurs aberrantes (âge, revenus, taux d’intérêt, dettes, etc.)

- Correction des incohérences par client (données mensuelles répétées)

- Suppression des colonnes non pertinentes (ID, nom, SSN)

- Encodage des variables catégorielles (Target Encoding)

- Normalisation des variables numériques

- Analyse de corrélation et suppression de variables redondantes

- Gestion avancée des valeurs manquantes

# Modèles testés

Les modèles suivants ont été implémentés et comparés :

Modèle	Accuracy
Perceptron	52.7 %
Naive Bayes (from scratch)	65 %
Réseau de neurones (MLPClassifier)	68 %
Arbre de décision	72 %
K-Nearest Neighbors (KNN)	76.5 % 

KNN a été retenu comme meilleur modèle, après optimisation du paramètre k.

# Évaluation

- Accuracy

- Precision / Recall / F1-score (macro & weighted)

- Matrices de confusion

- Courbes d’apprentissage

- Analyse du biais et de la variance

# Conclusion

Le modèle KNN offre le meilleur compromis entre performance et robustesse pour ce problème.
Ce projet démontre une maîtrise complète d’un projet de Machine Learning appliqué à la finance.

# Technologies utilisées

- Python

- Pandas, NumPy

- Scikit-learn

- Matplotlib / Seaborn

- Jupyter Notebook

  
