# Séance 4 : Régression et Classification Linéaires - Théorie et Pratique

## Vue d'ensemble

Cette quatrième séance approfondit les concepts de régression et classification linéaires avec une approche théorique et pratique. Nous allons explorer les mathématiques derrière ces modèles, les implémenter from scratch en Python, puis utiliser des bibliothèques standards comme scikit-learn. Cette séance inclut des notebooks Jupyter interactifs que vous pouvez exécuter localement pour expérimenter avec les concepts.

**Durée : 2 heures**

**Prérequis :**
- Concepts de la séance 3 (classification, régression, fonction objectif)
- Connaissances de base en Python
- Numpy, Matplotlib (installation nécessaire)

---

## Structure de la séance

```{important}
**Organisation :**

**Partie 1 (60 min) : Régression Linéaire**
- Théorie mathématique
- Implémentation from scratch
- Utilisation de scikit-learn
- Application en neurosciences

**Partie 2 (60 min) : Classification Linéaire**
- Régression logistique
- Implémentation from scratch
- Utilisation de scikit-learn
- Application en neurosciences
```

---

## Matériel pratique

### Notebooks Jupyter disponibles

```{admonition} Notebooks à télécharger
:class: tip

Pour suivre cette séance de manière interactive, téléchargez les notebooks Jupyter :

1. **[4.1_regression_lineaire.ipynb](notebooks/4.1_regression_lineaire.ipynb)** - Régression linéaire en pratique
2. **[4.2_classification_lineaire.ipynb](notebooks/4.2_classification_lineaire.ipynb)** - Classification linéaire en pratique
3. **[4.3_exercices.ipynb](notebooks/4.3_exercices.ipynb)** - Exercices pratiques

Tous les notebooks sont disponibles dans le dossier `seance4/notebooks/` du dépôt GitHub.
```

### Installation de l'environnement

```bash
# Installer les bibliothèques nécessaires
pip install numpy matplotlib scikit-learn jupyter

# Lancer Jupyter Notebook
jupyter notebook
```

---

## Partie 1 : Régression Linéaire

### 1.1 Rappel conceptuel

Dans la séance 3, nous avons vu que la régression linéaire est un modèle qui prédit une **valeur numérique continue** à partir de données d'entrée.

```{important}
**Formule de base :**

y = Wx + b

Où :
- **x** : variable d'entrée (feature)
- **W** : poids (weight) - paramètre à apprendre
- **b** : biais (bias) - paramètre à apprendre
- **y** : prédiction (sortie)
```

**Représentation géométrique :** Le modèle est une ligne (en 2D) ou un hyperplan (en dimensions supérieures) qui "s'ajuste" aux données.

### 1.2 Formulation mathématique générale

#### Cas multi-variables (régression linéaire multiple)

Lorsqu'on a plusieurs variables d'entrée (features), la formulation devient :

```{important}
**Forme matricielle :**

ŷ = Xw + b

Où :
- **X** : matrice de données (n échantillons × m features)
- **w** : vecteur de poids (m × 1)
- **b** : scalaire (biais)
- **ŷ** : vecteur de prédictions (n × 1)

Plus explicitement :
ŷᵢ = w₁x_{i1} + w₂x_{i2} + ... + wₘx_{im} + b
```

**Notation compacte :** On peut inclure le biais dans les poids en ajoutant une colonne de 1 à X :

```
ŷ = Xw  (où X inclut une colonne de 1, et w inclut b)
```

### 1.3 Fonction de coût (Loss Function)

Pour entraîner le modèle, nous devons quantifier l'erreur entre les prédictions et les vraies valeurs.

```{important}
**Mean Squared Error (MSE) :**

L(w, b) = (1/n) Σᵢ₌₁ⁿ (yᵢ - ŷᵢ)²

L(w, b) = (1/n) Σᵢ₌₁ⁿ (yᵢ - (Xᵢw + b))²

Où :
- n : nombre d'échantillons
- yᵢ : vraie valeur pour l'échantillon i
- ŷᵢ : prédiction pour l'échantillon i
```

**Objectif de l'entraînement :** Trouver les valeurs de w et b qui **minimisent** L(w, b).

### 1.4 Méthodes d'optimisation

Il existe deux approches principales pour trouver les paramètres optimaux :

#### A. Solution analytique (Équations normales)

Pour la régression linéaire, il existe une solution en forme fermée (closed-form solution) :

```{important}
**Équations normales :**

w* = (XᵀX)⁻¹Xᵀy

Où :
- Xᵀ : transposée de X
- (XᵀX)⁻¹ : inverse de XᵀX
- w* : poids optimaux
```

**Avantages :**
- ✅ Solution exacte
- ✅ Pas besoin d'itérations

**Inconvénients :**
- ❌ Coûteux en calcul pour grandes matrices (O(n³))
- ❌ Problèmes si XᵀX n'est pas inversible
- ❌ Ne fonctionne que pour la régression linéaire

#### B. Descente de gradient (Gradient Descent)

Méthode itérative qui ajuste progressivement les paramètres dans la direction qui réduit l'erreur.

```{important}
**Algorithme de descente de gradient :**

Répéter jusqu'à convergence :
    w := w - α × ∂L/∂w
    b := b - α × ∂L/∂b

Où :
- α : taux d'apprentissage (learning rate)
- ∂L/∂w : gradient (dérivée partielle) de L par rapport à w
- ∂L/∂b : gradient de L par rapport à b
```

**Calcul des gradients pour MSE :**

```
∂L/∂w = -(2/n) Σᵢ (yᵢ - ŷᵢ) xᵢ = -(2/n) Xᵀ(y - ŷ)
∂L/∂b = -(2/n) Σᵢ (yᵢ - ŷᵢ)
```

**Avantages :**
- ✅ Fonctionne pour de grandes données
- ✅ Généralisable à d'autres modèles
- ✅ Contrôle sur le processus d'optimisation

**Inconvénients :**
- ❌ Nécessite le choix du learning rate α
- ❌ Peut converger lentement
- ❌ Solution approximative

### 1.5 Implémentation pratique

```{admonition} Notebook pratique
:class: tip
**Ouvrez maintenant le notebook [4.1_regression_lineaire.ipynb](notebooks/4.1_regression_lineaire.ipynb)**

Dans ce notebook, vous allez :
1. Créer un dataset synthétique simple
2. Implémenter la régression linéaire from scratch avec NumPy
3. Visualiser la ligne de régression et la courbe d'apprentissage
4. Comparer avec l'implémentation de scikit-learn
5. Appliquer à un exemple de neurosciences (prédiction de vitesse)
```

### 1.6 Évaluation du modèle

Plusieurs métriques permettent d'évaluer la performance d'un modèle de régression :

#### A. Mean Squared Error (MSE)

```{important}
MSE = (1/n) Σᵢ (yᵢ - ŷᵢ)²

- Plus le MSE est **petit**, meilleur est le modèle
- Sensible aux valeurs aberrantes (outliers)
- Même unité que y²
```

#### B. Root Mean Squared Error (RMSE)

```{important}
RMSE = √MSE = √[(1/n) Σᵢ (yᵢ - ŷᵢ)²]

- Même unité que y
- Plus interprétable que MSE
```

#### C. Mean Absolute Error (MAE)

```{important}
MAE = (1/n) Σᵢ |yᵢ - ŷᵢ|

- Moins sensible aux outliers que MSE
- Même unité que y
```

#### D. Coefficient de détermination (R²)

```{important}
R² = 1 - (Σᵢ (yᵢ - ŷᵢ)²) / (Σᵢ (yᵢ - ȳ)²)

Où ȳ est la moyenne de y.

**Interprétation :**
- R² = 1 : modèle parfait
- R² = 0 : modèle aussi bon qu'une simple moyenne
- R² < 0 : modèle pire qu'une moyenne
- Généralement : 0 ≤ R² ≤ 1

**R² représente la proportion de variance expliquée par le modèle.**
```

### 1.7 Visualisation et diagnostic

```{admonition} Visualisations importantes
:class: note

**1. Scatter plot avec ligne de régression**
- Points de données vs ligne prédite
- Permet de voir visuellement l'ajustement

**2. Résidus (erreurs)**
- Plot des résidus vs prédictions
- Doit montrer une distribution aléatoire autour de 0
- Patterns indiquent des problèmes (non-linéarité, hétéroscédasticité)

**3. Courbe d'apprentissage**
- Évolution du loss pendant l'entraînement
- Doit diminuer et converger

**4. Prédictions vs vraies valeurs**
- Scatter plot : ŷ vs y
- Points doivent être proches de la ligne y=x
```

### 1.8 Exemple appliqué : Neurosciences

**Problème :** Prédire la vitesse de mouvement d'un singe à partir de l'activité de neurones dans le cortex moteur.

**Données :**
- **Entrées (X)** : Activité de N neurones (firing rates)
- **Sortie (y)** : Vitesse de mouvement (cm/s)
- **Échantillons** : Mesures à différents moments

**Processus :**
1. Enregistrer l'activité neuronale et la vitesse simultanément
2. Diviser les données en ensembles d'entraînement et de test
3. Entraîner le modèle de régression linéaire
4. Évaluer les prédictions sur l'ensemble de test
5. Interpréter les poids pour comprendre quelle activité neuronale prédit la vitesse

```{admonition} À explorer dans le notebook
:class: tip
Le notebook 4.1 contient un exemple complet avec des données simulées de neurosciences.
```

---

## Partie 2 : Classification Linéaire (Régression Logistique)

### 2.1 De la régression à la classification

La régression linéaire prédit des valeurs continues. Pour la classification, nous voulons prédire des **catégories discrètes**.

**Question :** Comment adapter la régression linéaire pour la classification ?

**Réponse :** Utiliser une **fonction d'activation** qui transforme les sorties continues en probabilités de classes.

### 2.2 La régression logistique

Malgré son nom, la régression logistique est un modèle de **classification**.

```{important}
**Architecture de la régression logistique :**

1. Combinaison linéaire : z = Wx + b
2. Fonction sigmoïde : ŷ = σ(z) = 1 / (1 + e⁻ᶻ)

Résultat final :
ŷ = σ(Wx + b) = 1 / (1 + e⁻⁽ᵂˣ⁺ᵇ⁾)

Où ŷ représente la **probabilité** que l'échantillon appartienne à la classe 1.
```

### 2.3 La fonction sigmoïde

```{important}
**Propriétés de la fonction sigmoïde σ(z) :**

σ(z) = 1 / (1 + e⁻ᶻ)

**Caractéristiques :**
- Sortie entre 0 et 1 → peut être interprétée comme une probabilité
- σ(0) = 0.5
- σ(z) → 1 quand z → +∞
- σ(z) → 0 quand z → -∞
- Forme en "S" (sigmoïde)
```

**Visualisation :**
```
  1.0 |           ___________
      |         /
  0.5 |        /
      |       /
  0.0 |______/
      |_____|_____|_____|_____|
        -5    0     5    10
```

**Décision de classification :**
```python
if ŷ ≥ 0.5:  # Probabilité ≥ 50%
    classe = 1
else:
    classe = 0
```

### 2.4 Fonction de coût : Binary Cross-Entropy

Pour la classification, on ne peut pas utiliser MSE (problème de non-convexité). On utilise la **log-loss** (binary cross-entropy).

```{important}
**Binary Cross-Entropy Loss :**

L(w, b) = -(1/n) Σᵢ [yᵢ log(ŷᵢ) + (1-yᵢ) log(1-ŷᵢ)]

Où :
- yᵢ ∈ {0, 1} : vraie classe
- ŷᵢ ∈ [0, 1] : probabilité prédite de la classe 1

**Interprétation :**
- Si yᵢ = 1 : L diminue quand ŷᵢ → 1 (bonne prédiction)
- Si yᵢ = 0 : L diminue quand ŷᵢ → 0 (bonne prédiction)
- Pénalise fortement les prédictions très confiantes mais fausses
```

**Pourquoi cette fonction ?**
- Dérivée de la théorie de l'information
- Convexe → garantit un optimum global
- Bien adaptée aux probabilités

### 2.5 Descente de gradient pour la régression logistique

```{important}
**Calcul des gradients :**

∂L/∂w = (1/n) Xᵀ(ŷ - y)
∂L/∂b = (1/n) Σᵢ (ŷᵢ - yᵢ)

**Mise à jour des paramètres :**

w := w - α × ∂L/∂w
b := b - α × ∂L/∂b

Note : La forme des gradients est similaire à la régression linéaire, mais ŷ = σ(Wx + b) ici.
```

### 2.6 Classification multi-classe

Pour plus de 2 classes, on utilise la **régression logistique multinomiale** (softmax regression).

```{important}
**Softmax (généralisation de sigmoïde) :**

Pour K classes, la probabilité de la classe k est :

ŷₖ = exp(zₖ) / Σⱼ exp(zⱼ)

Où zₖ = Wₖx + bₖ pour chaque classe k.

**Propriétés :**
- Σₖ ŷₖ = 1 (somme des probabilités = 1)
- Chaque ŷₖ ∈ [0, 1]
- La classe prédite est : argmax(ŷₖ)
```

**Fonction de coût : Categorical Cross-Entropy**

```
L = -(1/n) Σᵢ Σₖ yᵢₖ log(ŷᵢₖ)

Où yᵢₖ = 1 si l'échantillon i appartient à la classe k, 0 sinon.
```

### 2.7 Implémentation pratique

```{admonition} Notebook pratique
:class: tip
**Ouvrez maintenant le notebook [4.2_classification_lineaire.ipynb](notebooks/4.2_classification_lineaire.ipynb)**

Dans ce notebook, vous allez :
1. Créer un dataset 2D avec 2 classes linéairement séparables
2. Implémenter la régression logistique from scratch
3. Visualiser la frontière de décision
4. Comparer avec sklearn.linear_model.LogisticRegression
5. Étendre à la classification multi-classe
6. Appliquer à un exemple de neurosciences (classification de directions)
```

### 2.8 Géométrie de la classification linéaire

```{important}
**Frontière de décision (decision boundary) :**

La frontière de décision est l'ensemble des points où ŷ = 0.5, c'est-à-dire où :

Wx + b = 0

C'est une **ligne** (en 2D) ou un **hyperplan** (en dimensions supérieures) qui sépare l'espace en deux régions :
- Région 1 : Wx + b > 0 → classe 1
- Région 0 : Wx + b < 0 → classe 0
```

**Visualisation en 2D :**

Avec deux features x₁ et x₂, la frontière est :
```
w₁x₁ + w₂x₂ + b = 0
```

Cette ligne sépare les deux classes dans le plan (x₁, x₂).

### 2.9 Évaluation du modèle de classification

Plusieurs métriques sont utilisées pour évaluer la performance d'un classificateur :

#### A. Accuracy (Exactitude)

```{important}
Accuracy = (Nombre de prédictions correctes) / (Nombre total de prédictions)

Accuracy = (TP + TN) / (TP + TN + FP + FN)

Où :
- TP (True Positives) : Vrais positifs
- TN (True Negatives) : Vrais négatifs
- FP (False Positives) : Faux positifs
- FN (False Negatives) : Faux négatifs
```

**Limitation :** Peut être trompeuse avec des classes déséquilibrées.

#### B. Matrice de confusion

```{important}
**Structure de la matrice de confusion (2 classes) :**

                  Prédit Négatif    Prédit Positif
Réel Négatif           TN                FP
Réel Positif           FN                TP

**Matrice idéale :** Diagonale pleine, reste à zéro.
```

Pour K classes, c'est une matrice K×K où l'élément (i,j) indique combien d'échantillons de la classe i ont été classés comme classe j.

#### C. Précision et Rappel

```{important}
**Précision (Precision) :**
Proportion de prédictions positives qui sont correctes

Precision = TP / (TP + FP)

**Rappel (Recall / Sensibilité) :**
Proportion de vrais positifs correctement identifiés

Recall = TP / (TP + FN)

**F1-Score :**
Moyenne harmonique de précision et rappel

F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

#### D. Courbe ROC et AUC

```{admonition} ROC (Receiver Operating Characteristic)
:class: note

**Courbe ROC :**
- Axe X : Taux de faux positifs (FPR) = FP / (FP + TN)
- Axe Y : Taux de vrais positifs (TPR) = TP / (TP + FN) = Recall
- Montre le compromis entre TPR et FPR pour différents seuils

**AUC (Area Under Curve) :**
- Aire sous la courbe ROC
- AUC = 1 : classificateur parfait
- AUC = 0.5 : classificateur aléatoire
- Plus l'AUC est proche de 1, meilleur est le modèle
```

### 2.10 Visualisations pour la classification

```{admonition} Visualisations importantes
:class: note

**1. Frontière de décision**
- Visualisation 2D avec les points de données
- Montre comment le modèle sépare les classes
- Utile pour comprendre le comportement du modèle

**2. Matrice de confusion**
- Heatmap pour voir les erreurs de classification
- Diagonale = bonnes prédictions
- Hors diagonale = erreurs

**3. Courbe d'apprentissage**
- Évolution de la loss pendant l'entraînement
- Doit diminuer et converger

**4. Distributions des probabilités prédites**
- Histogrammes des ŷ pour chaque classe
- Bonnes séparation = distributions bien séparées
```

### 2.11 Exemple appliqué : Neurosciences

**Problème :** Classifier la direction du mouvement d'un singe (8 directions) à partir de l'activité de neurones dans le cortex moteur.

**Données :**
- **Entrées (X)** : Activité de N neurones (firing rates)
- **Sortie (y)** : Direction (0-7, représentant 8 directions)
- **Échantillons** : Enregistrements de multiples mouvements

**Approche :**
1. Classification multi-classe (8 classes)
2. Utiliser softmax regression
3. Évaluer avec matrice de confusion
4. Interpréter les poids pour comprendre quels neurones sont informatifs pour chaque direction

**Questions d'interprétation :**
- Quelles directions sont plus facilement distinguables ?
- Quels neurones contribuent le plus à la classification ?
- Y a-t-il des confusions systématiques entre certaines directions ?

```{admonition} À explorer dans le notebook
:class: tip
Le notebook 4.2 contient un exemple complet avec des données simulées de classification de directions de mouvement.
```

---

## Partie 3 : Comparaison et Choix du Modèle

### 3.1 Régression vs Classification : Tableau récapitulatif

```{important}
| Aspect | Régression Linéaire | Classification Linéaire |
|--------|-------------------|------------------------|
| **Type de sortie** | Continue (ℝ) | Discrète (catégories) |
| **Fonction d'activation** | Identité (y = z) | Sigmoïde (2 classes) ou Softmax (multi-classes) |
| **Fonction de coût** | MSE | Cross-Entropy |
| **Sortie du modèle** | Valeur prédite | Probabilités de classes |
| **Évaluation** | R², MSE, RMSE, MAE | Accuracy, Precision, Recall, F1, AUC |
| **Visualisation** | Ligne/plan de régression | Frontière de décision |
| **Applications** | Prédire vitesse, température, prix | Classifier espèces, diagnostics, directions |
```

### 3.2 Quand utiliser quel modèle ?

```{admonition} Régression Linéaire
:class: tip

**Utiliser quand :**
- ✅ La sortie est une valeur numérique continue
- ✅ La relation entre entrées et sortie est approximativement linéaire
- ✅ Vous voulez prédire une quantité

**Exemples :**
- Prédire la vitesse de mouvement
- Estimer l'âge à partir de données
- Prédire la température
- Estimer le prix d'une maison
```

```{admonition} Classification Linéaire
:class: tip

**Utiliser quand :**
- ✅ La sortie est une catégorie (binaire ou multi-classe)
- ✅ Les classes sont approximativement linéairement séparables
- ✅ Vous voulez assigner des labels

**Exemples :**
- Classifier la direction de mouvement
- Diagnostiquer maladie vs sain
- Reconnaître des chiffres manuscrits (simple)
- Spam vs non-spam
```

### 3.3 Limitations des modèles linéaires

```{admonition} Limitations importantes
:class: warning

**Les modèles linéaires ont des limites :**

1. **Hypothèse de linéarité**
   - Ne peuvent capturer que des relations linéaires
   - Performances médiocres si la vraie relation est non-linéaire

2. **Classes non linéairement séparables**
   - Classification linéaire échoue si les classes ne peuvent être séparées par une ligne/hyperplan
   - Exemple : problème XOR

3. **Sensibilité aux outliers**
   - Les valeurs aberrantes peuvent fortement influencer le modèle
   - Particulièrement vrai pour MSE (à cause du carré)

4. **Pas d'interactions complexes**
   - Ne peut pas capturer automatiquement les interactions entre features
   - Nécessite feature engineering manuel
```

**Solution :** Utiliser des modèles non-linéaires (réseaux de neurones, arbres de décision, SVM avec kernel, etc.)

### 3.4 Améliorer les modèles linéaires

```{admonition} Techniques d'amélioration
:class: note

**1. Feature Engineering**
- Ajouter des features polynomiales : x₁², x₁x₂, etc.
- Transformations : log(x), √x, etc.
- Features d'interaction

**2. Régularisation**
- Ridge (L2) : pénalise les grands poids
- Lasso (L1) : favorise la sparsité (sélection de features)
- Elastic Net : combinaison de L1 et L2

**3. Normalisation/Standardisation des données**
- Centrer et réduire les features
- Améliore la convergence de la descente de gradient
- Important quand les features ont des échelles différentes

**4. Validation croisée**
- Évaluer la généralisation du modèle
- Détecter le sur-apprentissage (overfitting)
```

---

## Concepts clés à retenir

```{important}
**Points essentiels de cette séance :**

**Régression Linéaire :**
1. Modèle : y = Wx + b (prédiction de valeurs continues)
2. Loss : MSE = (1/n) Σ(y - ŷ)²
3. Optimisation : Équations normales ou descente de gradient
4. Évaluation : R², MSE, RMSE, MAE
5. Visualisation : ligne de régression, résidus

**Classification Linéaire :**
1. Modèle : ŷ = σ(Wx + b) avec σ = sigmoïde
2. Loss : Binary Cross-Entropy
3. Optimisation : Descente de gradient
4. Évaluation : Accuracy, Precision, Recall, F1, AUC
5. Visualisation : frontière de décision, matrice de confusion

**Multi-classe :**
- Extension avec softmax
- Categorical cross-entropy
- Matrice de confusion K×K

**Pratique :**
- Implémentation from scratch pour comprendre
- Bibliothèques (sklearn) pour la production
- Toujours visualiser et valider
```

---

## Exercices pratiques

```{admonition} Notebook d'exercices
:class: tip
**Ouvrez le notebook [4.3_exercices.ipynb](notebooks/4.3_exercices.ipynb)**

Ce notebook contient des exercices progressifs pour mettre en pratique les concepts :

**Exercice 1 : Régression sur données réelles**
- Charger un dataset (par exemple, Boston Housing)
- Explorer les données
- Entraîner un modèle de régression
- Évaluer et interpréter les résultats

**Exercice 2 : Classification binaire**
- Dataset de neurosciences simulé
- Classifier deux types de mouvements
- Visualiser la frontière de décision
- Calculer les métriques de performance

**Exercice 3 : Classification multi-classe**
- Classifier 8 directions de mouvement
- Analyser la matrice de confusion
- Identifier les confusions communes
- Interpréter les poids du modèle

**Exercice 4 : Comparaison des approches**
- Implémenter from scratch vs sklearn
- Comparer les performances
- Analyser les différences

**Exercice 5 (Bonus) : Feature engineering**
- Améliorer la performance avec des features polynomiales
- Comparer modèle linéaire vs polynomial
```

---

## Ressources complémentaires

### Documentation et tutoriels

```{admonition} Ressources en ligne
:class: tip

**Documentation scikit-learn :**
- [Linear Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)
- [Logistic Regression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [User Guide - Linear Models](https://scikit-learn.org/stable/modules/linear_model.html)

**Tutoriels interactifs :**
- [Machine Learning Crash Course (Google)](https://developers.google.com/machine-learning/crash-course)
- [Scikit-learn Tutorials](https://scikit-learn.org/stable/tutorial/index.html)

**Visualisations interactives :**
- [TensorFlow Playground](https://playground.tensorflow.org/) - Pour visualiser les frontières de décision
- [Seeing Theory](https://seeing-theory.brown.edu/) - Pour la statistique visuelle
```

### Lectures approfondies

```{admonition} Pour aller plus loin
:class: note

**Livres recommandés :**
1. **"Pattern Recognition and Machine Learning"** - Christopher Bishop
   - Chapitre 3 : Linear Models for Regression
   - Chapitre 4 : Linear Models for Classification

2. **"The Elements of Statistical Learning"** - Hastie, Tibshirani, Friedman
   - Chapitre 3 : Linear Methods for Regression
   - Chapitre 4 : Linear Methods for Classification

3. **"Hands-On Machine Learning"** - Aurélien Géron
   - Chapitre 4 : Training Linear Models
   - Très pratique avec code Python
```

---

## Prochaines étapes

```{important}
**Pour la prochaine séance :**

Nous explorerons :
1. **Réseaux de neurones** : au-delà des modèles linéaires
2. **Apprentissage profond** : architectures multi-couches
3. **Rétropropagation** : comment entraîner des réseaux profonds
4. **Applications avancées** : vision par ordinateur, traitement du langage

**Préparez-vous en :**
- Complétant les exercices de cette séance
- Explorant les notebooks en profondeur
- Réfléchissant aux limitations des modèles linéaires
- Pensant à des problèmes non-linéaires en psychologie/neurosciences
```

---

## Récapitulatif de la séance

Aujourd'hui, nous avons :

✅ Compris la théorie mathématique de la régression et classification linéaires
✅ Implémenté ces modèles from scratch en Python
✅ Utilisé scikit-learn pour des implémentations robustes
✅ Appliqué ces techniques à des problèmes de neurosciences
✅ Appris à évaluer et visualiser les performances des modèles
✅ Identifié les limites des modèles linéaires

**Bravo pour votre travail ! La pratique est essentielle - n'hésitez pas à expérimenter avec les notebooks.**

---

**Fin de la Séance 4**
