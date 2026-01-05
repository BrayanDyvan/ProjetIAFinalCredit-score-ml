import numpy as np
import pandas as pd


class GaussianNaiveBayes:
    def fit(self, X, y):
        """
        Entraîne le modèle en calculant pour chaque classe :
          - La moyenne de chaque feature.
          - La variance de chaque feature (avec une petite constante pour éviter la division par zéro).
          - La probabilité a priori de la classe.
        """
        self.classes = np.unique(y)
        n_features = X.shape[1]
        self.mean = {}
        self.var = {}
        self.priors = {}
        for c in self.classes:
            X_c = X[y == c]
            self.mean[c] = np.mean(X_c, axis=0)
            self.var[c] = np.var(X_c, axis=0) + 1e-9  # petite constante pour la stabilité numérique
            self.priors[c] = X_c.shape[0] / X.shape[0]

    def gaussian_pdf(self, x, mean, var):
        """
        Calcule la densité de probabilité selon une loi normale pour la valeur x,
        donnée la moyenne et la variance.
        """
        coeff = 1.0 / np.sqrt(2.0 * np.pi * var)
        exponent = np.exp(-((x - mean) ** 2) / (2 * var))
        return coeff * exponent

    def predict(self, X):
        """
        Prédit la classe de chaque observation de X.
        Pour chaque observation, calcule la probabilité que cette observation appartienne à chaque classe
        et retourne la classe ayant le plus grand score.
        """
        y_pred = []
        for x in X:
            posteriors = []
            # Pour chaque classe, calcule la vraisemblance (les densités de probabilités produit)
            for c in self.classes:
                # Calcul du produit des probabilités pour chaque feature
                likelihood = np.prod(self.gaussian_pdf(x, self.mean[c], self.var[c]))
                posterior = likelihood * self.priors[c]
                posteriors.append(posterior)
            # La classe avec le posterior maximum est prédite.
            y_pred.append(self.classes[np.argmax(posteriors)])
        return np.array(y_pred)
