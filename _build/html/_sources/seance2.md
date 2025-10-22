# Séance 2 : Introduction à l'Apprentissage Automatique

## Vue d'ensemble

Cette deuxième séance explore la distinction fondamentale entre l'intelligence artificielle basée sur des règles explicites et l'apprentissage automatique. Nous découvrons les types d'IA, comment l'apprentissage automatique diffère de la programmation traditionnelle, et ses applications en psychologie et neurosciences.

---

## 1. Les types d'IA : Comprendre la hiérarchie

### 1.1 La relation entre IA, Machine Learning et Deep Learning

Nous entendons différents termes lorsqu'on parle d'IA : intelligence artificielle, apprentissage automatique (machine learning), apprentissage profond (deep learning), et plus récemment, IA générative. Comment ces concepts sont-ils liés ?

```{important}
**La hiérarchie des concepts :**

**IA (Intelligence Artificielle)** - Le domaine le plus large
↓
**Machine Learning (Apprentissage Automatique)** - Un sous-ensemble de l'IA
↓
**Deep Learning (Apprentissage Profond)** - Un sous-ensemble du ML
↓
**IA Générative** - Un sous-ensemble du DL
```

### 1.2 Comprendre la relation

```{admonition} Relations entre les concepts
:class: note
- **Machine Learning est un sous-ensemble de l'IA** : L'apprentissage automatique est l'une des approches possibles de l'IA, mais aujourd'hui c'est l'approche la plus populaire et la plus performante.

- **Deep Learning est un sous-ensemble du ML** : Il existe différentes approches de ML, mais le deep learning est une approche particulière qui est très populaire aujourd'hui dans le développement de l'IA.

- **Chaque terme a un sens distinct** : Bien que liés, ces quatre termes renvoient à des concepts différents avec des niveaux de spécificité différents.
```

**Pourquoi cette distinction est-elle importante ?**

Dans le domaine de l'IA, il existe différentes approches. Le machine learning est l'approche la plus réussie et performante actuellement. De même, dans le machine learning, le deep learning est l'approche qui a permis les plus grandes avancées récentes.

---

## 2. Programmation Traditionnelle vs Apprentissage Automatique

### 2.1 Qu'est-ce que la programmation traditionnelle ?

La **programmation traditionnelle** utilise une **approche basée sur des règles** ("rule-based approach").

```{admonition} Exemples de programmes traditionnels
:class: tip
Les applications que vous utilisez quotidiennement sont principalement développées avec de la programmation traditionnelle :
- Applications de téléphone portable
- Systèmes de courriel
- Systèmes d'exploitation (Windows, macOS, iOS, Android)
- Studium et autres plateformes éducatives
```

**Question fondamentale :**
Quelle est la différence entre ces systèmes et l'IA ? Les deux fonctionnent sur des machines et effectuent des tâches, mais quelle est la différence ?

### 2.2 L'approche basée sur des règles

Dans la programmation traditionnelle, le développeur crée des **règles explicites** qui déterminent le comportement du programme.

**Structure de base :**
```
IF X:
    DO Y
ELSE:
    DO Z
```

Ces règles "if-then" (si-alors) sont prédéfinies par le programmeur. Selon la complexité du système désiré, vous aurez besoin de créer un programme avec plusieurs, voire des milliers de ces règles.

```{important}
**Caractéristique clé de la programmation traditionnelle :**
La performance du système dépend entièrement de l'exhaustivité et de la précision des règles définies par le programmeur.
```

### 2.3 L'approche de l'apprentissage automatique

L'**apprentissage automatique (Machine Learning)** fonctionne différemment : **apprentissage à partir de données et d'exemples, pas de règles prédéfinies**.

```{admonition} Définition formelle (Russell and Norvig, 2021)
:class: note
L'apprentissage automatique est un sous-domaine de l'IA qui étudie la capacité à améliorer les performances sur la base de l'expérience.
```

**Interprétation importante :**
"L'expérience" dans cette définition fait référence aux **données** que nous utilisons pour entraîner notre modèle d'IA.

```{admonition} Citation de Stanford HAI
:class: tip
"Les humains programment les machines pour qu'elles se comportent de manière intelligente, comme par exemple jouer aux échecs, mais aujourd'hui nous mettons l'accent sur les machines capables d'apprendre, au moins « un peu » comme le font les êtres humains."

Source: Stanford Human-Centered AI Institute
```

---

## 3. Exemples Comparatifs : Règles vs Apprentissage

### 3.1 Exemple 1 : Détection de spam dans les courriels

#### Approche traditionnelle (basée sur des règles)

**Principe :** Le programmeur définit des règles explicites pour classer un courriel comme spam.

**Exemple de code Python :**
```python
def est_spam(texte_email):
    mots_clefs_spam = ["gagner", "gratuit", "offre", "cliquez maintenant"]
    for mot in mots_clefs_spam:
        if mot in texte_email.lower():
            return True
    return False

courriel = "Félicitations ! Vous gagnez un iPhone gratuit. Cliquez maintenant !"
if est_spam(courriel):
    print("Ceci est un e-mail spam.")
else:
    print("Ceci n'est pas un e-mail spam.")
```

**Comment ça fonctionne :**
1. On définit une liste de mots-clés qui apparaissent souvent dans les messages de spam
2. Le programme vérifie si le courriel contient ces mots-clés
3. Si l'un de ces mots est présent, il classe le courriel comme spam

**La performance de ce programme dépend des règles et de leur degré d'exhaustivité et de précision.**

```{admonition} Limitations de l'approche basée sur règles
:class: warning
**Inflexibilité :** Incapable de s'adapter aux nouveaux modèles de spam. Les développeurs de spam pourraient simplement changer les mots dans leurs messages.

**Faux positifs :** Risque de mal classer des courriels légitimes contenant des mots similaires (par exemple : "Parking gratuit disponible").

**Maintenance constante :** Nécessite une mise à jour manuelle constante des règles.
```

#### Approche apprentissage automatique

**Principe :** Il n'y a pas de règles ou de mots-clés prédéfinis. Le modèle voit de nombreux exemples de courriels spam et non-spam et apprend automatiquement les motifs.

**Exemple de code Python (simplifié) :**
```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Données d'entraînement
courriels = [
    "Gagnez des vacances gratuites maintenant", 
    "Réunion demain à 15h", 
    "Offre spéciale juste pour vous", 
    "Déjeuner avec l'équipe"
]
etiquettes = [1, 0, 1, 0]  # 1 = spam, 0 = non spam

# Convertir le texte en caractéristiques numériques
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(courriels)

# Entraîner un modèle de détection de spam
model = MultinomialNB()
model.fit(X, etiquettes)
```

```{admonition} Note importante
:class: note
C'est juste un exemple pour montrer la différence d'approche. Les détails techniques ne sont pas importants ici. Ce qui compte, c'est de comprendre que les deux méthodes sont implémentées dans un cadre de programmation, mais les programmes contiennent des choses complètement différentes.
```

**Comment ça fonctionne :**
Le modèle apprend à partir des données d'entraînement à détecter des motifs, tels que des combinaisons de mots ou des fréquences, qui indiquent la présence de spam.

**Les motifs découverts par l'apprentissage automatique sont plus complexes et subtils que la simple liste de mots-clés de l'approche traditionnelle.**

```{admonition} Avantages de l'approche ML
:class: tip
✅ **Plus flexible et adaptatif** : Peut s'adapter à de nouveaux types de spam avec un entraînement supplémentaire

✅ **Meilleure précision** : Avec des ensembles de données larges et diversifiés

✅ **Découvre des patterns complexes** : Au-delà des mots-clés simples
```

```{admonition} Défis de l'approche ML
:class: warning
⚠️ **Nécessite des données étiquetées** : Besoin d'exemples annotés de spam et non-spam

⚠️ **Ressources informatiques** : Plus gourmand en calcul que les règles simples

⚠️ **Erreurs difficiles à interpréter** : Moins transparent que des règles explicites
```

---

### 3.2 Exemple 2 : Système de feux de circulation

#### Approche basée sur des règles explicites

**Principe :** Un système de feux de circulation programmé avec un ensemble simple de règles.

**Exemple de règle :**
"Passer au vert pendant 30 secondes dans une direction, puis passer au rouge et laisser passer l'autre direction pendant 30 secondes."

**Approche plus complexe :**
Vous pourriez définir des durées différentes selon les heures de la journée :
- Passer au vert pendant 60 secondes de 8h à 9h du matin (heure de pointe)
- Passer au vert pendant 30 secondes le reste de la journée

```{admonition} Limitation
:class: warning
Cette méthode est **rigide** et ne tient pas compte des variations réelles des conditions de circulation (congestion imprévisible, accidents, événements spéciaux).
```

#### Approche apprentissage automatique

**Principe :** Un système basé sur l'apprentissage automatique pourrait utiliser des données de trafic en temps réel.

**Fonctionnement :**
- **Sources de données :** Capteurs, caméras de surveillance du trafic
- **Ajustement dynamique :** Les durées des feux s'adaptent selon le trafic à chaque moment
- **Apprentissage continu :** Le système apprend des motifs qui facilitent la circulation et réduisent les accidents

```{important}
Le système s'adapte automatiquement sans programmation manuelle des règles. Il découvre les patterns optimaux à partir des données de trafic réelles.
```

---

### 3.3 Exemple 3 : Jeu d'échecs

#### Approche basée sur des règles explicites

**Les premiers programmes d'échecs :**
Les développeurs programmaient des stratégies et des mouvements spécifiques basés sur des règles "if-then" :
- "Si votre adversaire déplace sa reine, déplacez votre cavalier pour le bloquer."
- "Si vous pouvez capturer un pion sans risque, faites-le."

```{admonition} Limitation
:class: warning
Ces programmes ne pouvaient effectuer qu'un nombre limité de mouvements prédéfinis et ne pouvaient pas s'adapter à de nouvelles stratégies. Si le joueur change sa stratégie, le système n'est pas capable de s'adapter parce que les règles sont fixes.
```

#### Cas historique : Deep Blue (1997)

```{admonition} Deep Blue vs Garry Kasparov
:class: note
**Fait intéressant :** L'IA Deep Blue qui a battu Garry Kasparov en 1997 utilisait ce type de programmation traditionnelle (aussi connue sous le nom d'IA symbolique).

**Fonctionnement :**
- Contenait un grand arbre de mouvements et de stratégies possibles
- Pouvait déterminer le meilleur mouvement à chaque état du jeu
- Structure : "Si le jeu est dans cette situation, il y a 3 choses que ton compétiteur pourrait faire. IF elle décide de faire X, DO A. IF elle décide de faire Y, DO B, etc."

**Limitation :** L'arbre était fixe, non flexible, non adaptatif. Il ne pouvait donc pas apprendre une nouvelle stratégie.
```

#### Approche apprentissage automatique : AlphaZero

**AlphaZero (DeepMind)** utilise une approche complètement différente basée sur l'apprentissage automatique (deep learning).

```{important}
**Révolution majeure :**
AlphaZero a appris à jouer aux échecs en **jouant des millions de parties contre lui-même**, en améliorant ses stratégies et en prenant des décisions **sans règles prédéfinies**.

Il a découvert des stratégies gagnantes que même les humains n'avaient pas envisagées, en se basant uniquement sur les données (l'historique de ses parties).
```

**Différence clé :**
- **Il n'y a aucune règle prédéfinie** qui détermine les mouvements à chaque moment
- **Tout est découvert à partir de données**
- Il est très probable qu'AlphaZero découvre et utilise des stratégies qui n'ont pas été découvertes par les humains

```{admonition} Témoignage de joueurs professionnels
:class: tip
Les joueurs professionnels ont affirmé avoir vu des stratégies ou des mouvements utilisés par AlphaZero qu'ils n'avaient jamais vus auparavant. L'IA a littéralement inventé de nouvelles façons de jouer !
```

---

### 3.4 Exemple 4 : Reconnaissance des émotions à partir des expressions faciales

#### Approche basée sur des règles explicites

**Principe :** Vous définissez des règles explicites pour reconnaître les émotions.

**Exemples de règles :**
- "Si les coins de la bouche sont relevés et que les dents sont visibles → classer comme un sourire (joie)"
- "Si les sourcils sont froncés et que la bouche est tournée vers le bas → classer comme une moue (tristesse)"

**Le système repose entièrement sur des descriptions prédéfinies de configurations faciales.**

```{admonition} Limitations
:class: warning
- Difficulté à capturer la subtilité des expressions humaines
- Incapacité à gérer les variations individuelles et culturelles
- Nécessite de définir manuellement chaque configuration possible
- Ne peut pas s'adapter à de nouvelles expressions ou contextes
```

#### Approche basée sur l'apprentissage automatique

**Principe :** On entraîne un modèle sur un ensemble de données d'expressions faciales étiquetées.

**Fonctionnement :**
- L'IA apprend les motifs qui correspondent aux différentes émotions
- **Sans nécessiter de règles explicites**
- Le modèle découvre automatiquement les caractéristiques pertinentes

```{admonition} Avantages
:class: tip
- Capture des patterns subtils et complexes
- S'adapte aux variations individuelles
- Peut améliorer ses performances avec plus de données
- Découvre des associations que nous n'aurions pas pensé à programmer
```

---

## 4. Qu'est-ce que l'Apprentissage Automatique ?

### 4.1 Définition formelle

```{important}
**Apprentissage automatique (Machine Learning) :**
Algorithmes permettant aux ordinateurs d'apprendre des formes (ou motifs; patterns en anglais) à partir de données.
```

**Concept clé :** Au lieu de programmer explicitement chaque règle, nous laissons l'ordinateur découvrir les patterns dans les données.

### 4.2 Les deux grandes catégories

Il existe deux catégories principales d'apprentissage automatique :
1. **Apprentissage supervisé**
2. **Apprentissage non supervisé**

---

## 5. Apprentissage Supervisé

### 5.1 Définition et principe

```{admonition} Apprentissage supervisé
:class: note
**Apprentissage à partir de données annotées**

Pour l'apprentissage supervisé, nous avons besoin de données annotées, c'est-à-dire que pour chaque échantillon de données, nous avons besoin de son **étiquette** ou de ses **annotations** - les choses que nous voulons que le modèle apprenne à prédire à partir de ces données.
```

**Le terme "supervisé" :**
Nous "supervisons" l'apprentissage du modèle en lui fournissant les bonnes réponses (étiquettes) pour chaque exemple.

### 5.2 Exemple : Détection des émotions dans les images

**Objectif :** Entraîner un modèle de ML pour détecter les expressions du visage.

**Ce dont nous avons besoin :**
- **Données d'entrée (Input) :** Images de visages
- **Données de sortie (Output) / Étiquettes :** Les émotions correspondantes (joie, tristesse, colère, etc.)

```{important}
Pour chaque image de visage, nous devons fournir l'étiquette de l'émotion exprimée. C'est ce qu'on appelle des **données annotées**.
```

### 5.3 Exemple : Détection de visages

**Objectif :** Entraîner un modèle capable de détecter des visages dans des images.

**Ce dont nous avons besoin :**
- **Données d'entrée :** Images diverses (avec et sans visages)
- **Étiquettes :** Pour chaque image, indiquer s'il y a un visage ou non (et possiblement sa localisation)

```{admonition} Pourquoi "supervisé" ?
:class: tip
C'est de l'apprentissage **supervisé** parce que pour chaque échantillon de données (dans ce cas, chaque image), nous avons besoin d'une **supervision explicite** fournie par des étiquettes.

Sans ces étiquettes, le modèle ne saurait pas ce qu'il doit apprendre à prédire.
```

### 5.4 Application en neurosciences : DeepLabCut

**DeepLabCut** est un modèle d'apprentissage automatique très utile et populaire pour analyser les mouvements en neurosciences et psychologie.

```{admonition} Qu'est-ce que DeepLabCut ?
:class: note
Un modèle d'apprentissage automatique capable de :
- Détecter les parties du corps des animaux
- Suivre leurs mouvements dans des vidéos
- Permettre aux chercheurs de suivre les mouvements de toutes les parties du corps avec une grande précision
```

**Applications concrètes :**

Les chercheurs qui s'intéressent aux mouvements des animaux ou des êtres humains peuvent :
1. Enregistrer les mouvements dans des vidéos
2. Utiliser DeepLabCut pour suivre automatiquement les mouvements de toutes les parties du corps
3. Analyser ces données pour comprendre la structure du mouvement et le contrôle du mouvement par le cerveau

**Exemples d'utilisation :**
- **Mouches (Fly) :** Processus de ponte (egg-laying process)
- **Rongeurs :** Comportements d'exploration
- **Primates :** Mouvements de préhension
- **Humains :** Analyse de la marche, sauts verticaux (jumping vertically)

**Comment ça fonctionne :**
1. Des vidéos d'animaux sont fournies au modèle
2. Le modèle est entraîné à détecter la position de chaque partie du corps dans chaque image des vidéos
3. Résultat : Vous pouvez suivre les parties du corps tout au long des vidéos avec une grande précision

```{important}
**Très utile pour :**
- Étudier le mouvement en neurosciences et en psychologie
- Comprendre la structure du mouvement
- Comprendre le contrôle du mouvement par le cerveau
```

---

## 6. Apprentissage Non Supervisé

### 6.1 Définition et principe

```{admonition} Apprentissage non supervisé
:class: note
**Apprentissage à partir de données non annotées**

Pour l'apprentissage non supervisé, **aucune étiquette n'est nécessaire**. Les algorithmes peuvent fonctionner avec des données non annotées pour découvrir des motifs (patterns) dans les données.
```

**Différence fondamentale :**
Nous ne supervisons pas exactement ce qui doit être appris à partir des données. On donne au modèle les données et **on le laisse découvrir les choses lui-même**.

### 6.2 Principe du clustering (regroupement)

La plupart des modèles non supervisés **regroupent les échantillons de données en fonction de leur similarité**.

**Objectif :**
Créer des groupes (ou **clusters**) où :
1. Les échantillons **à l'intérieur** d'un groupe sont **plus similaires entre eux**
2. Les échantillons sont **différents** des échantillons des **autres groupes**

```{admonition} Point important
:class: tip
Comment vous définissez la **mesure de similarité** influencera le regroupement ou clustering. La plupart des recherches dans le domaine de l'apprentissage non supervisé portent en fait sur les mesures de similarité.
```

### 6.3 Exemple : Regroupement de visages

**Scénario :** Vous avez une base de données d'images de visages sans étiquettes.

**Ce qu'un modèle d'apprentissage non supervisé peut faire :**
Regrouper automatiquement les images simplement selon leurs similarités, sans connaître l'identité des personnes ou leurs caractéristiques.

**Résultat potentiel :**
- Groupe 1 : Visages d'enfants
- Groupe 2 : Visages d'adultes
- Groupe 3 : Visages souriants
- Groupe 4 : Visages de profil
- etc.

Le modèle découvre ces groupements naturellement, sans qu'on lui dise à l'avance quelles catégories chercher.

### 6.4 Application en neurosciences : MoSeq

**MoSeq (Motion Sequencing algorithm)** est un exemple d'apprentissage non supervisé pour l'analyse comportementale.

```{admonition} Qu'est-ce que MoSeq ?
:class: note
**Une méthode objective pour décrire le répertoire comportemental de la souris (ou d'autres animaux)**

MoSeq découvre automatiquement les différents comportements dans des vidéos d'animaux sans avoir besoin d'étiquettes prédéfinies.
```

**Le problème à résoudre :**
1. Nous avons enregistré des vidéos de souris (ou d'autres animaux) bougeant dans un environnement
2. Nous souhaitons découvrir les segments des vidéos contenant des **comportements similaires**
3. Nous voulons segmenter les vidéos en segments qui contiennent :
   - Des comportements **similaires à l'intérieur** de ce segment
   - Des comportements **différents** des comportements dans les autres segments

**Comment MoSeq fonctionne :**

Le modèle est capable de :
- Détecter automatiquement les différents comportements de la souris dans la vidéo
- Nous donner les parties de la vidéo qui correspondent à chaque comportement
- Créer un "dictionnaire" objectif des comportements

**Généralité de l'approche :**
Ce n'est pas seulement pour les souris ! Vous pouvez l'utiliser pour regrouper le comportement de n'importe quel animal, y compris les humains.

```{admonition} Quand utiliser l'apprentissage non supervisé ?
:class: tip
Ce type d'apprentissage automatique est très utile lorsque :

1. **Vous avez beaucoup de données non annotées** et il serait difficile ou coûteux d'annoter tous les échantillons

2. **Vous ne savez pas exactement quels patterns chercher** - vous voulez que le modèle découvre lui-même les structures dans les données

3. **Vous voulez une analyse objective** sans biais humains dans l'étiquetage
```

---

## 7. Applications en Psychologie et Neurosciences

### 7.1 Reconstruction des états visuels du cerveau : Peut-on lire dans les pensées ?

Une des applications les plus fascinantes de l'apprentissage automatique est la capacité de **reconstruire ce qu'une personne voit ou imagine** à partir de l'activité cérébrale.

```{admonition} La question
:class: question
**Peut-on lire le contenu visuel de la pensée humaine ?** (Mind reading en anglais)

C'est une question qui fascine les scientifiques depuis des années.
```

#### Comment ça fonctionne : L'approche de l'IA

**Type de modèle :** Apprentissage supervisé

**Structure du modèle :**

```{important}
**Entrées (Inputs) :** Données IRMf enregistrées du cerveau au moment où les humains regardent des images ou des vidéos

**Sorties (Outputs) :** Les images ou vidéos reconstruites que les humains ont regardées pendant l'enregistrement
```

**Processus d'entraînement :**

1. **Phase d'entraînement :**
   - Des participants regardent des images/vidéos
   - On enregistre leur activité cérébrale (IRMf)
   - Le modèle apprend les associations entre patterns d'activité cérébrale et contenu visuel

2. **Phase de test :**
   - On enregistre l'activité cérébrale d'une personne regardant une nouvelle image
   - Le modèle reconstruit l'image à partir de l'activité cérébrale seule

**Référence :** Takagi and Nishimoto, 2023
[Lien vers la recherche](https://sites.google.com/view/stablediffusion-with-brain/)

#### Applications potentielles

```{admonition} En quoi est-ce utile pour les neurosciences cognitives ?
:class: tip
**1. Comprendre le cerveau :**
Après avoir créé un tel modèle, nous pouvons l'examiner pour découvrir :
- Comment il pouvait reconstruire les images
- Quelle combinaison de motifs d'activité cérébrale permet la reconstruction
- Cela nous aide à mieux comprendre le fonctionnement du cerveau

**2. Accéder aux états non conscients :**
Nous pouvons l'utiliser pour reconstruire les contenus du cerveau dans des états non conscients :
- **Les rêves** : Que voyons-nous quand nous rêvons ?
- **Patients souffrant de troubles de la conscience** : Communication avec des patients en état végétatif ou minimalement conscients
- **Obtenir une meilleure compréhension des bases neurales** de ces états

**3. Applications cliniques potentielles :**
- Communication avec des patients qui ne peuvent pas parler
- Diagnostic de troubles neurologiques
- Prothèses visuelles pour les aveugles
```

---

## 8. Récapitulatif et Comparaison

### 8.1 Tableau comparatif : Règles vs Apprentissage Automatique

| Aspect | Programmation Traditionnelle (Règles) | Apprentissage Automatique |
|--------|---------------------------------------|---------------------------|
| **Base** | Règles explicites "if-then" | Patterns découverts dans les données |
| **Développement** | Programmeur définit toutes les règles | Modèle apprend automatiquement |
| **Flexibilité** | Rigide, fixe | Adaptatif, peut apprendre de nouvelles situations |
| **Performance** | Dépend de l'exhaustivité des règles | S'améliore avec plus de données |
| **Nouveaux cas** | Nécessite modification manuelle du code | S'adapte avec réentraînement |
| **Transparence** | Facile à comprendre (règles explicites) | Moins transparent ("boîte noire") |
| **Exemples** | Deep Blue (1997), systèmes de feux fixes | AlphaZero, détection de spam moderne, DeepLabCut |

### 8.2 Apprentissage Supervisé vs Non Supervisé

| Aspect | Apprentissage Supervisé | Apprentissage Non Supervisé |
|--------|------------------------|----------------------------|
| **Données requises** | Données annotées (avec étiquettes) | Données non annotées |
| **Supervision** | Explicite (étiquettes) | Aucune |
| **Objectif** | Prédire une sortie spécifique | Découvrir des structures cachées |
| **Coût** | Plus élevé (annotation nécessaire) | Moins élevé |
| **Exemples** | Détection d'émotions, DeepLabCut | Clustering de comportements, MoSeq |
| **Quand l'utiliser** | Quand on sait ce qu'on cherche | Quand on explore les données |

---

## 9. Concepts Clés à Retenir

```{important}
**Les points essentiels de la séance 2 :**

1. **Hiérarchie :** IA ⊃ Machine Learning ⊃ Deep Learning ⊃ IA Générative

2. **Deux paradigmes :**
   - **Programmation traditionnelle** : Règles explicites prédéfinies
   - **Apprentissage automatique** : Découverte de patterns à partir de données

3. **L'apprentissage automatique est plus :**
   - Flexible et adaptatif
   - Capable de découvrir des patterns complexes
   - Performant avec beaucoup de données

4. **Deux types d'apprentissage :**
   - **Supervisé** : Avec données annotées (exemples : détection d'émotions, DeepLabCut)
   - **Non supervisé** : Sans annotations (exemples : clustering, MoSeq)

5. **Applications en psychologie/neurosciences :**
   - Analyse de mouvements (DeepLabCut)
   - Analyse de comportements (MoSeq)
   - Reconstruction d'états visuels (IRMf + ML)
```

---

## 10. Questions de Réflexion

```{admonition} Questions pour la discussion
:class: question
1. **Quand préféreriez-vous utiliser des règles explicites plutôt que l'apprentissage automatique ?**
   - Y a-t-il des situations où les règles sont préférables ?

2. **Implications éthiques de la lecture des pensées :**
   - Si nous pouvons reconstruire ce qu'une personne voit, quelles sont les implications pour la vie privée ?
   - Comment protéger les pensées privées ?

3. **Objectivité vs Biais :**
   - L'apprentissage non supervisé est-il vraiment objectif ?
   - Les mesures de similarité ne reflètent-elles pas aussi des biais humains ?

4. **Applications en psychologie clinique :**
   - Comment l'apprentissage automatique pourrait-il transformer l'évaluation psychologique ?
   - Quels sont les risques de s'appuyer trop sur l'IA pour le diagnostic ?

5. **Transparence vs Performance :**
   - Est-ce un problème que les modèles d'apprentissage automatique soient des "boîtes noires" ?
   - Comment équilibrer performance et interprétabilité ?
```

---

## 11. Ressources Complémentaires

### Lectures recommandées

```{admonition} Références académiques
:class: note
- **Russell, S., & Norvig, P. (2021).** *Artificial Intelligence: A Modern Approach* (4th ed.)
  - Chapitre sur le Machine Learning

- **Takagi, Y., & Nishimoto, S. (2023).** *High-resolution image reconstruction with latent diffusion models from human brain activity*
  - [Lien vers le projet](https://sites.google.com/view/stablediffusion-with-brain/)
```

### Outils et démos

- **DeepLabCut :** [http://www.mackenzi elab.org/deeplabcut](http://www.mackenzielab.org/deeplabcut)
- **MoSeq :** Motion Sequencing pour l'analyse comportementale
- **Stanford HAI :** [https://hai.stanford.edu/](https://hai.stanford.edu/)

---

## 12. Préparation pour la Séance 3

La prochaine séance approfondira les concepts de l'apprentissage automatique :
- Comment entraîne-t-on un modèle concrètement ?
- Qu'est-ce que l'apprentissage profond (deep learning) ?
- Comment les réseaux de neurones artificiels fonctionnent-ils ?

```{admonition} Pour aller plus loin
:class: tip
Avant la prochaine séance, réfléchissez à :
- Comment le cerveau apprend-il de nouvelles choses ?
- Quelles sont les similarités et différences entre l'apprentissage humain et l'apprentissage automatique ?
- Comment pourriez-vous utiliser l'apprentissage automatique dans votre domaine de recherche en psychologie ?
```

---

## Notes pour l'Enseignant

```{admonition} Timing suggéré
:class: note
**Segment 1 : Programmation vs ML (45 minutes)**
- Introduction et hiérarchie des concepts (10 min)
- Exemples comparatifs (30 min)
  - Spam (10 min)
  - Feux de circulation (5 min)
  - Échecs (10 min)
  - Reconnaissance d'émotions (5 min)
- Discussion (5 min)

**Segment 2 : Types d'apprentissage (35 minutes)**
- Introduction au ML (5 min)
- Apprentissage supervisé + exemples (15 min)
  - DeepLabCut démo
- Apprentissage non supervisé + exemples (10 min)
  - MoSeq démo
- Application : Reconstruction visuelle (5 min)

**Discussion finale (10 minutes)**
```

### Points d'emphase

1. **Insister sur la différence fondamentale :**
   - Règles = programmées par humain
   - ML = découvertes dans les données

2. **Utiliser des exemples concrets :**
   - Les étudiants comprennent mieux avec des cas pratiques
   - Relier au quotidien (spam, etc.)

3. **Montrer l'utilité en psychologie/neurosciences :**
   - DeepLabCut : analyse de mouvement
   - MoSeq : comportements objectifs
   - Reconstruction visuelle : comprendre le cerveau

4. **Encourager la pensée critique :**
   - Quand utiliser quelle approche ?
   - Limitations et biais
   - Implications éthiques

### Activités suggérées

- **Démonstration interactive :** Montrer DeepLabCut et MoSeq en action
- **Exercice de réflexion :** Demander aux étudiants de proposer d'autres domaines où ML serait utile
- **Débat :** Lecture des pensées - fascinant ou inquiétant ?
