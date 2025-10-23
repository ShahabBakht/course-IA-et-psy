# Séance 3 : Classification et Régression

## Vue d'ensemble

Cette troisième séance approfondit les concepts d'apprentissage automatique en explorant les deux grandes familles d'apprentissage supervisé : la **classification** et la **régression**. Nous verrons comment ces approches sont appliquées en psychologie et neurosciences, notamment pour les neuroprothèses et la modélisation du cerveau. Nous comprendrons aussi les éléments clés de l'apprentissage automatique et comment les modèles sont entraînés.

---

## 1. Applications de l'apprentissage automatique en psychologie et neurosciences

### 1.1 Neuroprothèses : Décodage de la parole chez les patients paralysés

#### Le défi

Un des défis les plus importants et touchants de la neuroscience moderne est de redonner la parole aux personnes qui l'ont perdue à cause d'un accident neurologique ou d'un désordre neurologique, comme dans le cas de la paralysie.

```{admonition} Vidéo recommandée
:class: tip
Pour comprendre visuellement ce défi et les progrès réalisés, visionnez cette vidéo :
[A high-performance neuroprosthesis for speech decoding and avatar control](https://www.youtube.com/watch?v=vL7yMn6kiMg)
```

#### Les résultats actuels

**Performance actuelle :**
- Vitesse de décodage : **78 mots par minute**
- Conversation naturelle : **160 mots par minute**

```{important}
C'est un grand progrès, mais nous avons encore une marge de progression significative. L'objectif est de se rapprocher le plus possible de la vitesse de conversation naturelle pour permettre une communication fluide.
```

#### Comment le modèle d'apprentissage automatique fonctionne

```{admonition} Architecture du système
:class: note

**Entrée → Modèle → Sortie**

- **Entrée :** Données cérébrales (activité neuronale enregistrée)
- **Modèle :** Système d'apprentissage automatique entraîné
- **Sortie :** Parole générée et texte correspondant
```

**Le modèle reçoit les données du cerveau et génère ce qui nous intéresse : la parole et les textes correspondants.**

#### Le défi de la collecte de données

Une question cruciale se pose : **Comment collecter les données nécessaires pour entraîner le modèle alors que les patients ne peuvent pas parler ?**

```{admonition} Solution ingénieuse
:class: tip

**Technique de collecte de données :**

Avant d'entraîner le modèle, les chercheurs ont :

1. Présenté de nombreuses phrases et textes à l'écran au patient
2. Demandé au patient d'**essayer** de prononcer ces phrases
3. Pendant que le patient essayait (sans y arriver physiquement), enregistré l'activité cérébrale
4. Supposé que ces activités correspondent aux phrases visées

**Résultat :** Phrases et activité cérébrale - tout ce qui est nécessaire pour entraîner le modèle !
```

Cette technique fonctionne remarquablement bien, comme on peut le voir dans la vidéo.

#### L'importance de la qualité des données

```{important}
**Leçons clés :**

1. **Le patient est le juge ultime** : Tant que le patient est satisfait des résultats du modèle (le modèle dit ce que le patient veut dire), le modèle fonctionne. Si le patient voudrait dire quelque chose mais le modèle génère autre chose, c'est une erreur.

2. **La collecte de données est cruciale** : Trouver la meilleure façon de collecter les données nécessaires est une partie très importante de tous les projets d'apprentissage automatique.

3. **Qualité = Performance** : La performance de votre modèle est largement influencée par la qualité de vos données.
```

### 1.2 Neuroprothèses : Neuralink

#### Le projet Neuralink

Un autre exemple de neuroprothèse provient de Neuralink, l'entreprise dirigée par Elon Musk.

```{admonition} Vidéo démonstration
:class: tip
Regardez cette démonstration célèbre :
[Pager, a nine year old Macaque, plays MindPong with his Neuralink](https://www.youtube.com/watch?v=rsCul1sp4hQ)

Certains d'entre vous l'ont peut-être déjà vue grâce à leur publicité intensive !
```

#### Le défi

**Objectif :** Contrôler quelque chose avec l'activité cérébrale, par exemple :
- Jouer au jeu de pong
- Bouger un objet sur l'écran
- **Uniquement en y pensant, sans interface manuelle**

#### Le rôle de l'apprentissage automatique

```{admonition} Comment ça fonctionne
:class: note

**Architecture similaire au décodage de la parole :**

**Activité cérébrale (entrée)** → **Modèle d'apprentissage automatique** → **Commandes de mouvement (sortie)**

Le modèle apprend à traduire les intentions de mouvement (activité cérébrale) en commandes concrètes pour contrôler le curseur ou le jeu.
```

### 1.3 Modélisation du cerveau utilisant l'apprentissage automatique

#### Un nouveau paradigme de recherche

Au-delà des applications cliniques, l'apprentissage automatique, et particulièrement les **réseaux de neurones artificiels**, ouvre de nouvelles possibilités pour la recherche fondamentale en neurosciences.

```{admonition} Lecture recommandée
:class: tip
Pour approfondir ce paradigme de recherche, consultez cet article de Nature Reviews Neuroscience :

[The neuroconnectionist research program](https://www.nature.com/articles/s41583-023-00705-w)
```

#### Le principe

**Question de recherche :** Est-il possible de développer un modèle capable de reproduire les activités des neurones du cerveau en réponse aux mêmes stimuli ?

**Exemple concret :** Créer un modèle qui peut imiter les activités du système visuel en réponse à une série de stimuli visuels.

```{admonition} Architecture du modèle
:class: note

**Stimuli visuels (entrée)** → **Modèle de neurones artificiels** → **Activités neuronales prédites (sortie)**

Le modèle est entraîné à générer des activités neuronales qui correspondent aux activités réelles mesurées dans le cerveau.
```

**Pourquoi est-ce important ?**

Cette approche permet de :
1. **Tester des hypothèses** sur le fonctionnement du cerveau
2. **Comprendre les représentations** dans différentes régions cérébrales
3. **Prédire** les réponses neuronales à de nouveaux stimuli
4. **Développer des théories computationnelles** de la cognition

---

## 2. Les éléments clés de l'apprentissage automatique

Maintenant que nous avons vu plusieurs exemples d'applications, explorons les composantes fondamentales qui constituent tout système d'apprentissage automatique.

### 2.1 Les données d'entraînement

```{important}
**Définition :**
Les données d'entraînement sont les **données d'entrée et de sortie à partir desquelles le modèle apprend**.
```

**Dans tous nos exemples précédents :**
- Décodage de la parole : Activité cérébrale (entrée) + Parole correspondante (sortie)
- Neuralink : Activité cérébrale (entrée) + Mouvements désirés (sortie)
- Modélisation du cerveau : Stimuli visuels (entrée) + Activité neuronale (sortie)

```{admonition} Point critique
:class: warning
Sans données appropriées et de qualité, il est impossible d'entraîner un modèle performant. La collecte et la préparation des données représentent souvent la majeure partie du travail dans un projet d'apprentissage automatique.
```

### 2.2 Le modèle paramétrique de l'apprentissage automatique

```{important}
**Définition :**
Le modèle est le **type de structure paramétrique** qui va apprendre à partir des données.
```

#### Questions importantes pour les chercheurs et développeurs

**Choix du modèle :** Quel type de modèle est le plus approprié pour ce cas ?

**Caractéristiques des modèles :**
- Les modèles sont **paramétrés** : ils ont des paramètres ajustables
- Ces paramètres permettent aux modèles de **s'adapter à chaque tâche**
- Le processus d'apprentissage consiste à ajuster ces paramètres

```{admonition} Exemples de types de modèles
:class: note
- Régression linéaire (modèle simple)
- Réseaux de neurones artificiels (modèles complexes)
- Arbres de décision
- Support Vector Machines (SVM)
- Réseaux de neurones profonds (Deep Learning)
```

### 2.3 L'objectif de l'apprentissage (fonction objectif)

```{important}
**Définition :**
La fonction objectif est la **fonction mathématique qui détermine le but ou l'objectif de l'apprentissage**.
```

**Rôle :** Elle quantifie à quel point le modèle performe bien sur les données d'entraînement.

```{admonition} Principe fondamental
:class: tip
Dans chaque application, on a besoin de **définir très précisément** la fonction objectif du modèle. Cette fonction guide le processus d'apprentissage en indiquant ce que le modèle doit optimiser.
```

### 2.4 L'algorithme d'entraînement (algorithme d'optimisation)

```{important}
**Définition :**
L'algorithme d'entraînement est le **processus qui ajuste les paramètres du modèle** pour améliorer sa performance selon la fonction objectif.
```

**Comment ça fonctionne :**
1. L'algorithme évalue la performance actuelle du modèle (via la fonction objectif)
2. Il détermine comment modifier les paramètres pour améliorer cette performance
3. Il ajuste les paramètres en conséquence
4. Le processus se répète jusqu'à convergence

---

## 3. Types de données en apprentissage automatique

### 3.1 La diversité des types de données

L'une des forces de l'apprentissage automatique moderne est sa capacité à traiter une grande variété de types de données.

```{admonition} Types de données courantes
:class: note

**Données structurées :**
- Images
- Vidéos
- Textes
- Audio/Sons
- Séries temporelles

**Données scientifiques spécialisées :**
- Activités cérébrales (EEG, fMRI, enregistrements neuronaux)
- Données génomiques
- Données médicales (radiographies, IRM, etc.)

**Données multi-modales :**
- Images + Textes (captions)
- Vidéos + Sons
- Combinaisons diverses
```

```{important}
**Principe général :**
Toutes les données possibles qui peuvent être collectées de n'importe quelle manière peuvent potentiellement être utilisées pour entraîner des modèles d'apprentissage automatique.
```

### 3.2 L'apprentissage multi-modal

On peut aussi utiliser un **mélange de types de données** comme entrées d'un modèle.

**Exemples :**
- Images avec des textes qui décrivent le contenu (captions)
- Vidéos et leurs bandes sonores
- Textes et métadonnées temporelles

Cette approche multi-modale est de plus en plus utilisée dans les systèmes modernes d'IA.

---

## 4. L'apprentissage supervisé : Classification

### 4.1 Introduction à la classification

```{important}
**Définition :**
La classification est un type d'apprentissage supervisé où le modèle doit **assigner chaque échantillon à une catégorie prédéfinie**.
```

En fonction des sorties souhaitées du modèle, on peut avoir deux types différents d'apprentissage supervisé : la **classification** et la **régression**.

```{admonition} Point clé
:class: note
Les **entrées** de ces deux types de problèmes ne sont pas différentes. Le même type d'entrées peut être utilisé pour la régression et la classification. C'est le **type de sortie** qui détermine s'il s'agit de classification ou de régression.
```

### 4.2 Exemple illustratif : Chiens et chats

Considérons des images de chiens et de chats comme données d'entrée.

**Pour la classification :**
On peut demander au modèle de déterminer s'il y a un chat ou un chien dans chaque image → **Classification**

**Pour la régression :**
On peut demander au modèle d'estimer l'âge du chat ou du chien dans chaque image → **Régression**

```{admonition} Question de réflexion
:class: tip
Qu'est-ce qu'on pourrait demander d'autre à un modèle avec ces données d'entrée ?

Exemples possibles :
- Identifier la race de l'animal (classification multi-classe)
- Estimer le poids de l'animal (régression)
- Déterminer si l'animal est à l'intérieur ou à l'extérieur (classification)
```

### 4.3 Les sorties catégorielles

```{important}
**Caractéristique des sorties de classification :**

Les sorties catégorielles **classent les échantillons dans des catégories qualitatives**.

Il existe un **nombre fini** de catégories possibles pour chaque échantillon de données.
```

**Représentation numérique :**
Les sorties catégorielles sont représentées par des nombres entiers :
- 2 classes : 0, 1
- 3 classes : 0, 1, 2
- N classes : 0, 1, 2, ..., N-1

**Exemple binaire (chiens vs chats) :**
```
['chat', 'chat', 'chat', 'chien', 'chat', …, 'chien', 'chat']
```
Devient :
```
[0, 0, 0, 1, 0, ..., 1, 0]
```

Le modèle doit apprendre à générer le bon numéro pour chaque image en fonction de sa catégorie.

### 4.4 Exemple en neurosciences : Classification des directions de mouvement

#### Contexte expérimental

**Situation :** Un singe bouge sa main dans 7 directions différentes :
- Gauche
- Droite
- Haut
- Bas
- Haut-gauche
- Haut-droite
- Bas-gauche
- Bas-droite

**Mesure :** En même temps, on enregistre les activités d'une région du cerveau, par exemple le **cortex moteur**.

**Données obtenues :** Mouvements de la main + Activité cérébrale correspondante

#### Définir le problème de classification

```{admonition} Question
:class: tip
Comment peut-on définir un problème d'apprentissage automatique de classification à partir de ces données ?
```

**Réponse :**

**Entrées :** Activité cérébrale (enregistrements neuronaux)
**Sorties :** Direction du mouvement (0-7, une pour chaque direction)
**Objectif :** Le modèle doit apprendre à prédire la direction du mouvement à partir de l'activité cérébrale

```python
# Représentation des directions
directions = ['gauche', 'droite', 'haut', 'bas', 
              'haut-gauche', 'haut-droite', 'bas-gauche', 'bas-droite']
# Encodées comme : [0, 1, 2, 3, 4, 5, 6, 7]
```

#### Applications de ce modèle

```{admonition} Utilité du modèle entraîné
:class: note

**1. Application clinique (neuroprothèse) :**
Déterminer les mouvements possibles à partir de l'activité du cerveau lorsque l'animal (ou un patient) **ne peut pas bouger sa main**, c'est-à-dire lorsqu'il est paralysé.

**2. Recherche fondamentale (décodage du cerveau) :**
Déterminer si ce type de classification est possible à partir des activités d'une région du cerveau. Est-ce que l'information nécessaire pour effectuer cette classification existe dans cette région ?

**Résultats connus :**
- ✅ Cortex moteur : Oui, la classification est possible (l'information est présente)
- ❌ Cortex visuel : Non, l'information nécessaire n'est pas disponible dans cette région
```

Cette méthode scientifique s'appelle le **décodage du cerveau** (brain decoding).

### 4.5 Exemple en neurosciences : Classification d'orientation visuelle

#### Contexte expérimental

**Situation :** Une souris observe des stimuli visuels, en l'occurrence des **motifs en noir et blanc** (gratings) à des angles différents :
- Horizontal (0°)
- Vertical (90°)
- Oblique à 45°
- Oblique à 135°
- Etc.

**Mesure :** En même temps, on enregistre les activités d'une région du cerveau, par exemple le **cortex visuel**.

```{admonition} Question
:class: tip
Comment peut-on définir un problème d'apprentissage automatique de classification à partir de ces données ?
```

#### La structure des données

**Représentation matricielle :**

La matrice enregistrée présente toutes les données d'activité. Chaque élément de cette matrice montre la **réponse d'un neurone** à un **stimulus spécifique** à un **angle spécifique**.

```
         Neurone 1  Neurone 2  Neurone 3  ...  Neurone N
Essai 1     0.43      0.87       0.12    ...    0.65
Essai 2     0.51      0.79       0.21    ...    0.71
Essai 3     0.38      0.92       0.08    ...    0.58
...
```

**Entrées :** Toutes les données enregistrées sur les neurones (vecteur d'activité de tous les neurones)
**Sorties :** Angle correspondant (0°, 45°, 90°, 135°, etc.)

#### Application du modèle

```{important}
Après avoir formé ce modèle, que pourrait-il faire ?

**Réponse :** C'est un exemple très simplifié de **"mind reading"** !

À partir de l'activité neuronale dans le cortex visuel, le modèle peut déterminer quel stimulus visuel (quelle orientation) l'animal est en train de regarder.
```

Cette technique a des implications profondes pour :
- Comprendre comment le cerveau encode l'information visuelle
- Développer des interfaces cerveau-machine
- Décoder les perceptions visuelles

### 4.6 La géométrie de la classification

#### Visualisation simplifiée avec deux neurones

Considérons une version simplifiée de l'exemple de classification des directions de mouvement :
- Seulement **deux directions** : gauche et droite
- Seulement **deux neurones** enregistrés

```{admonition} Représentation graphique
:class: note

**Diagramme de dispersion (scatter plot) :**

Axe X : Activité du neurone 1
Axe Y : Activité du neurone 2

- ⚪ Cercles : Mouvements vers la gauche
- 🔺 Triangles : Mouvements vers la droite

Chaque point représente les activités des deux neurones pour l'une des deux directions.
```

**Objectif du modèle :** Apprendre à déterminer si les activités correspondent à un mouvement vers la gauche ou vers la droite.

#### Classification linéaire

```{important}
**Le modèle apprend une courbe qui sépare les échantillons des deux classes.**

Pour la **classification linéaire**, cette courbe est une **ligne droite**.
```

**Caractéristiques :**
- De chaque côté de cette ligne, il n'y a que les échantillons d'une seule classe
- C'est appelé "linéaire" parce que la courbe est une ligne
- Mathématiquement : une combinaison linéaire des activités des neurones détermine la classe

**Formule générale :**
```
y = w₁ × x₁ + w₂ × x₂ + b
```
Où :
- x₁, x₂ : activités des neurones 1 et 2
- w₁, w₂ : poids (paramètres à apprendre)
- b : biais (paramètre à apprendre)

#### Classification non linéaire

```{important}
On peut avoir une **classification non linéaire** avec une courbe qui n'est pas une ligne et qui est plus complexe.
```

**Avantages d'un modèle non linéaire :**
- Plus complexe et plus adaptatif
- Plus de paramètres à ajuster
- Capable d'une classification plus complexe
- Peut séparer des classes qui ne sont pas linéairement séparables

**Inconvénients :**
- Risque de surapprentissage (overfitting)
- Plus difficile à interpréter
- Nécessite plus de données d'entraînement

---

## 5. L'apprentissage supervisé : Régression

### 5.1 Introduction à la régression

```{important}
**Définition :**
La régression est un type d'apprentissage supervisé où le modèle doit **prédire une valeur numérique continue** pour chaque échantillon.
```

### 5.2 Différence entre classification et régression

**Comme mentionné précédemment, la différence réside uniquement dans les sorties :**

**Classification :**
- Sorties catégorielles
- Nombre fini de possibilités
- Exemples : chat/chien, directions de mouvement

**Régression :**
- Sorties numériques continues
- Peuvent prendre (théoriquement) toutes les valeurs possibles
- Exemples : âge, vitesse, température

```{admonition} Point clé
:class: note
C'est différent de la classification où vous avez des sorties catégorielles avec des possibilités limitées. En régression, les sorties peuvent varier de manière continue dans une plage de valeurs.
```

### 5.3 Exemple en neurosciences : Régression de la vitesse de mouvement

#### Contexte expérimental

**Situation :** Notre singe bouge sa main dans différentes directions et à différentes vitesses. En même temps, on enregistre les activités du cerveau.

```{admonition} Question
:class: tip
Pouvez-vous définir un problème de régression à partir de ces données ?
```

#### Définition du problème

**Entrées :** Activité cérébrale pendant les mouvements (mêmes données que pour la classification)
**Sorties :** Vitesse du mouvement (valeur numérique continue)

**Différence avec la classification :**
Au lieu de prédire la **direction** (catégorie), le modèle doit maintenant apprendre à estimer la **vitesse** (valeur numérique).

```{important}
Comme la vitesse peut prendre n'importe quelle valeur (dans une plage), il s'agit d'une **régression**.
```

**Représentation des sorties :**
```
[0.5, 1.2, 0.8, 2.1, 1.5, ..., 0.9, 1.7] (m/s)
```

#### Performance du modèle

Voici deux exemples de performances de modèles pour l'estimation de la vitesse à partir de données cérébrales, utilisant deux types de modèles :

**LSTM (Long Short-Term Memory)** et **SVR (Support Vector Regression)**

```{admonition} Visualisation de la performance
:class: note

Dans chaque diagramme :
- **Courbe noire** : Vitesse réelle (véritable)
- **Courbes verte/rouge** : Sorties du modèle (prédictions)

**Évaluation de la performance :**
- Plus les courbes sont similaires, plus le modèle est précis
- Les deux courbes doivent être aussi similaires que possible
- La différence entre les deux courbes doit être minimale
```

Les résultats montrent que les sorties des modèles sont très proches des valeurs réelles, démontrant l'efficacité de l'approche.

### 5.4 Exemple en neurosciences : Prédiction d'activité neuronale

#### Contexte expérimental

**Situation :** Un modèle capable de générer des activités du cerveau à partir de stimuli visuels.

```{admonition} Questions de réflexion
:class: tip
- Comment peut-on faire cela ?
- Quelles sont les données d'entrée ?
- Quelles sont les sorties ?
- Comment peut-on collecter ces données ?
```

#### Définition du problème

**Entrées :** Stimuli visuels (images)
**Sorties :** Activités neuronales prédites (valeurs numériques continues pour chaque neurone)

**Processus de collecte :**
1. Présenter des images à un animal (ou humain)
2. Enregistrer simultanément l'activité de neurones dans le cortex visuel
3. Obtenir des paires (image, activité neuronale)

**Application :**
Ce type de modèle permet de :
- Prédire comment le cerveau répondra à de nouveaux stimuli
- Tester des hypothèses sur le traitement visuel
- Comprendre les représentations neuronales

### 5.5 La géométrie de la régression

#### Visualisation avec un seul neurone

Considérons un exemple très simplifié basé sur l'estimation de la vitesse à partir de données du cerveau.

**Hypothèse simplificatrice :** On a seulement **un neurone** enregistré pendant les mouvements du singe.

**Objectif :** Entraîner le modèle pour estimer la vitesse à partir des activités de ce seul neurone.

```{admonition} Représentation graphique
:class: note

**Diagramme de dispersion :**

Axe X : Activité du neurone
Axe Y : Vitesse du mouvement

Les points montrent la relation entre l'activité neuronale et la vitesse observée.
```

#### Le modèle comme une courbe

```{important}
**Les courbes en rouge sont les modèles.**

Chaque courbe peut nous donner la vitesse pour chaque valeur d'activité de neurone.
```

**Deux types de modèles :**

1. **Modèle linéaire :**
   - La courbe est une ligne droite
   - Formule : `vitesse = w × activité + b`
   - Simple mais limité

2. **Modèle non linéaire :**
   - La courbe est plus complexe qu'une ligne
   - Peut capturer des relations plus subtiles
   - Plus de paramètres, plus adaptatif

```{important}
**Principe général :**

Le modèle est une **courbe** qu'on peut entraîner (ou optimiser) pour faire quelque chose, estimer quelque chose, etc.

Les paramètres du modèle déterminent la forme de cette courbe.
```

---

## 6. Entraînement du modèle : La fonction objectif

### 6.1 Introduction à la fonction objectif

Maintenant que nous comprenons ce qu'est un modèle et comment il peut faire des prédictions (classification ou régression), une question cruciale se pose :

```{admonition} Question centrale
:class: tip
Comment peut-on **quantifier** si un modèle fonctionne bien ou mal ?

Comment peut-on mesurer sa **performance** ?
```

**Réponse :** Nous avons besoin d'une **fonction objectif** (aussi appelée **fonction de perte** ou **loss function**).

### 6.2 Principe de la quantification de l'erreur

```{important}
Pour **quantifier la performance** d'un modèle, nous devons comparer :
- Les sorties **réelles** (les vraies valeurs)
- Les sorties **prédites** par le modèle

La **différence** entre ces deux valeurs constitue l'**erreur** du modèle.
```

### 6.3 Notation mathématique

**Convention de notation :**
- **y** : Sorties réelles (ground truth, vraies valeurs)
- **ŷ** (y chapeau) : Sorties estimées/prédites par le modèle

**Pour différents échantillons :**
- y₁, y₂, y₃, ... : vraies valeurs pour les échantillons 1, 2, 3, ...
- ŷ₁, ŷ₂, ŷ₃, ... : valeurs prédites pour les échantillons 1, 2, 3, ...

### 6.4 Mean Squared Error (MSE) - Erreur quadratique moyenne

L'une des fonctions objectif les plus courantes pour la régression est le **Mean Squared Error** (MSE).

```{important}
**Formule du MSE :**

MSE = (1/n) × Σ(yᵢ - ŷᵢ)²

Où :
- n : nombre d'échantillons
- yᵢ : valeur réelle pour l'échantillon i
- ŷᵢ : valeur prédite pour l'échantillon i
- Σ : somme sur tous les échantillons
```

**Calcul étape par étape :**

1. Pour chaque échantillon, prendre la sortie réelle (y) et la sortie estimée (ŷ)
2. Calculer la différence : (y - ŷ)
3. Calculer le carré de cette différence : (y - ŷ)²
4. Faire la moyenne sur tous les échantillons

```{admonition} Pourquoi utiliser le carré ?
:class: note

**Raisons d'utiliser (y - ŷ)² plutôt que |y - ŷ| :**

1. **Pénalise davantage les grandes erreurs** : Une erreur de 10 donne 100, alors qu'une erreur de 2 donne seulement 4
2. **Propriétés mathématiques avantageuses** : Le carré est différentiable partout, ce qui facilite l'optimisation
3. **Convention standard** : Permet de comparer facilement différents modèles
```

**Interprétation :**
- MSE = 0 : Modèle parfait (prédictions exactes)
- MSE petit : Bon modèle
- MSE grand : Modèle imprécis

---

## 7. Entraînement du modèle : L'algorithme d'optimisation

### 7.1 Le rôle de l'algorithme d'apprentissage

```{important}
Avec une quantification des erreurs (fonction objectif), maintenant on peut **entraîner le modèle**.

L'**algorithme d'apprentissage** (aka algorithme d'optimisation) utilise la sortie de la fonction objectif pour **changer, ajuster, ou optimiser les paramètres du modèle** (W).
```

**Objectif :** L'algorithme d'apprentissage modifie les paramètres de manière que :
- Le taux d'erreur du modèle soit **réduit**
- La performance du modèle **augmente, s'améliore** !

### 7.2 Une analogie : Le "Guessing Game"

Pour comprendre comment fonctionne l'algorithme d'apprentissage, considérons une analogie simple : le jeu de devinette.

#### Le jeu

**Règles du jeu :**
1. Je choisis un chiffre (par exemple, 5014)
2. Vous devez le deviner
3. À chaque essai, je vous indique :
   - La **direction** de votre erreur (trop haut ou trop bas)
   - La **magnitude** de votre erreur (à quelle distance vous êtes)

#### Parallèle avec l'apprentissage automatique

```{admonition} Correspondances
:class: note

**Dans le jeu :**
- Votre estimation = Sortie du modèle
- Le vrai chiffre = Vraie valeur (y)
- Mon feedback = Fonction objectif
- Comment vous ajustez votre estimation = Algorithme d'apprentissage
```

#### Le processus itératif

**Itération 1 :**
- Vous : "300" (estimation initiale aléatoire)
- Moi : "↑ Trop bas, erreur = 4714"

**Itération 2 :**
- Vous : "2000" (ajustement basé sur le feedback)
- Moi : "↑ Trop bas, erreur = 3014"

**Itération 3 :**
- Vous : "5000" (ajustement)
- Moi : "↑ Trop bas, erreur = 14"

**Itération 4 :**
- Vous : "5100" (ajustement)
- Moi : "↓ Trop haut, erreur = 86"

**Itération 5 :**
- Vous : "5050" (ajustement)
- Moi : "↓ Trop haut, erreur = 36"

**Itération 6 :**
- Vous : "5010" (ajustement)
- Moi : "↓ Trop haut, erreur = 4"

```{important}
**Observation clé :**

À chaque itération, l'**erreur diminue** (en général). C'est exactement ce qui se passe dans l'apprentissage automatique !
```

### 7.3 La courbe d'apprentissage

La **courbe d'apprentissage** montre l'évolution de l'erreur au cours du processus d'entraînement.

```{admonition} Caractéristiques d'une bonne courbe d'apprentissage
:class: tip

**On veut que cette courbe montre la réduction des erreurs au cours de la période d'apprentissage.**

Axe X : Nombre d'itérations (epochs)
Axe Y : Erreur (fonction objectif)

Une bonne courbe d'apprentissage :
- ↘ Décroissante
- Converge vers une valeur faible
- Stable (sans oscillations excessives)
```

```{admonition} Signes de problèmes
:class: warning

Si l'erreur **ne diminue pas** au cours de l'apprentissage, cela indique un problème :
- ❌ La quantification de l'erreur ne fonctionne pas
- ❌ L'algorithme d'apprentissage ne fonctionne pas
- ❌ Le modèle n'est pas adapté au problème
- ❌ Les données ne sont pas appropriées
```

### 7.4 Éléments clés du processus d'apprentissage

En revisitant notre analogie du "Guessing Game", on identifie les éléments essentiels :

```{important}
**Composantes de l'apprentissage :**

1. **Initialisation** : On commence par une estimation aléatoire
   - Sans feedback, on n'a aucune information
   - La première estimation est par hasard

2. **Feedback** : Les erreurs sont le feedback nécessaire
   - Indiquent la direction et la magnitude du changement nécessaire
   - Essentiels pour guider l'ajustement

3. **Ajustement** : Modification basée sur le feedback
   - Les paramètres sont modifiés pour réduire l'erreur
   - Processus itératif et progressif
```

---

## 8. Les paramètres du modèle : Métaphore des boutons ajustables

### 8.1 Visualisation des paramètres

```{important}
**Métaphore utile :**

Le modèle est comme un **appareil avec beaucoup de boutons ajustables** (knobs).

Les **paramètres du modèle** sont ces boutons ajustables.
```

**Rôle de l'algorithme d'apprentissage :**
L'algorithme doit changer les paramètres selon les erreurs du modèle, permettant au modèle d'améliorer sa performance.

### 8.2 Le processus d'ajustement

```{admonition} Processus d'optimisation
:class: note

**Étape 1 : Initialisation**
- Tous les paramètres sont ajustés à des valeurs **aléatoires**
- La première sortie du modèle est donc aléatoire

**Étape 2 : Itérations d'entraînement**
- Les paramètres sont ajustés à chaque itération
- Objectif : **Réduire ou minimiser la perte** (fonction objectif)

**Étape 3 : Convergence**
- Le processus continue jusqu'à ce que l'erreur soit minimale
- Ou jusqu'à ce qu'elle cesse de diminuer significativement
```

### 8.3 Exemple : Régression linéaire

Pour comprendre concrètement les paramètres, considérons le modèle de régression le plus simple : la **régression linéaire**.

#### Paramétrisation du modèle

```{important}
**Formule du modèle linéaire :**

y = W · x + b

Où :
- **x** : données d'entrée
- **W** : poids (weights) - PARAMÈTRE à apprendre
- **b** : biais (bias) - PARAMÈTRE à apprendre
- **y** : sortie du modèle
```

**Il y a deux paramètres ajustables :**
1. Les poids **W**
2. Les biais **b**

Ce sont ces paramètres qui sont ajustés pendant l'entraînement.

#### Visualisation géométrique : Le rôle des poids (W)

```{admonition} Impact du poids W
:class: note

Le modèle est représenté par une **ligne** qui détermine la sortie selon les entrées.

**Les poids changent la pente de la ligne.**

Pendant l'entraînement, la pente est ajustée pour trouver la **meilleure pente** pour ces données.
```

**Exemple : Estimation de la vitesse de mouvement**

Axe X : Activité neuronale
Axe Y : Vitesse prédite

Différentes valeurs de W donnent différentes pentes :
- W petit → ligne presque horizontale
- W moyen → pente modérée
- W grand → ligne très inclinée

#### Visualisation géométrique : Le rôle du biais (b)

```{admonition} Impact du biais b
:class: note

Le biais peut changer la **position** de la ligne (décalage vertical).

L'algorithme d'apprentissage l'ajuste pour obtenir le **meilleur ajustement** possible.
```

**Effet du biais :**
- b = 0 : la ligne passe par l'origine
- b > 0 : la ligne est décalée vers le haut
- b < 0 : la ligne est décalée vers le bas

```{important}
**Objectif de l'optimisation :**

Trouver les valeurs de W et b qui donnent la ligne qui **s'ajuste le mieux** aux données (minimise l'erreur quadratique moyenne).
```

### 8.4 Comment l'algorithme ajuste-t-il les paramètres ?

```{admonition} Aperçu (détails dans le cours d'apprentissage profond)
:class: note

**Question :** Comment l'algorithme d'apprentissage peut-il faire ça ?

**Réponse courte :** C'est un sujet que nous explorerons en détail dans le cours d'apprentissage profond.

**Idée générale :**
- L'algorithme calcule le **gradient** (dérivée) de la fonction objectif par rapport aux paramètres
- Ce gradient indique dans quelle direction changer chaque paramètre
- Les paramètres sont ajustés dans cette direction
- Le processus est répété itérativement
```

---

## 9. Concepts clés à retenir

```{important}
**Points essentiels de cette séance :**

1. **Applications en neurosciences** : L'apprentissage automatique permet des avancées majeures dans les neuroprothèses (décodage de la parole, contrôle par la pensée) et la modélisation du cerveau.

2. **Éléments clés de l'apprentissage automatique** :
   - Données d'entraînement (entrées + sorties)
   - Modèle paramétrique
   - Fonction objectif
   - Algorithme d'optimisation

3. **Classification vs Régression** :
   - **Classification** : sorties catégorielles (nombre fini de classes)
   - **Régression** : sorties numériques continues
   - La différence réside dans le type de sortie, pas dans les entrées

4. **Géométrie de l'apprentissage** :
   - **Classification** : trouver une courbe qui sépare les classes
   - **Régression** : trouver une courbe qui prédit les valeurs
   - Modèles linéaires vs non linéaires

5. **Entraînement du modèle** :
   - **Fonction objectif** : quantifie l'erreur (ex: MSE)
   - **Algorithme d'optimisation** : ajuste les paramètres pour réduire l'erreur
   - **Processus itératif** : amélioration progressive par feedback

6. **Paramètres du modèle** : boutons ajustables qui déterminent le comportement du modèle (ex: poids W et biais b dans la régression linéaire)

7. **Importance de la qualité des données** : la performance du modèle dépend crucialement de la qualité et de la pertinence des données d'entraînement
```

---

## 10. Questions de réflexion

```{admonition} Pour approfondir votre compréhension
:class: tip

1. **Applications cliniques** : Quelles autres applications neuroprothétiques pourrait-on développer avec l'apprentissage automatique ? Pensez à différentes fonctions cérébrales qui pourraient être restaurées.

2. **Éthique et neuroprothèses** : Quelles sont les implications éthiques du décodage de l'activité cérébrale ? Où tracer la ligne entre aide médicale et invasion de la vie privée mentale ?

3. **Classification vs régression** : Donnez trois exemples de problèmes de classification et trois exemples de problèmes de régression en psychologie.

4. **Qualité des données** : Pourquoi la qualité des données est-elle si cruciale ? Quels problèmes peuvent survenir avec des données biaisées ou de mauvaise qualité ?

5. **Modèles linéaires vs non linéaires** : Dans quelles situations un modèle linéaire serait-il suffisant ? Quand aurait-on absolument besoin d'un modèle non linéaire ?

6. **Fonction objectif** : Pourquoi utilise-t-on l'erreur quadratique (carrée) plutôt que l'erreur absolue ? Quels sont les avantages et inconvénients ?

7. **Décodage du cerveau** : Comment le décodage du cerveau peut-il aider à comprendre comment différentes régions cérébrales traitent l'information ?
```

---

## 11. Ressources complémentaires

### Articles scientifiques

```{admonition} Lectures recommandées
:class: tip

**Sur les neuroprothèses et le décodage de la parole :**
- Metzger et al. (2023). "A high-performance neuroprosthesis for speech decoding and avatar control." *Nature*. [doi: 10.1038/s41586-023-06443-4](https://doi.org/10.1038/s41586-023-06443-4)

**Sur la modélisation du cerveau avec l'apprentissage automatique :**
- Richards et al. (2023). "The neuroconnectionist research program." *Nature Reviews Neuroscience*. [Article disponible](https://www.nature.com/articles/s41583-023-00705-w)
```

### Vidéos

```{admonition} Démonstrations visuelles
:class: note

**Neuroprothèse de décodage de la parole :**
[Vidéo démonstration](https://www.youtube.com/watch?v=vL7yMn6kiMg)

**Neuralink - Contrôle par la pensée :**
[Pager joue à MindPong](https://www.youtube.com/watch?v=rsCul1sp4hQ)
```

### Concepts pour la prochaine séance

Dans la prochaine séance, nous approfondirons :
- Les différents types d'algorithmes d'apprentissage automatique
- L'apprentissage non supervisé
- Les réseaux de neurones et l'apprentissage profond
- Comment éviter le surapprentissage (overfitting)

---

**Fin de la Séance 3**
