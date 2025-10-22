# Séance 1 : Introduction à l'Intelligence Artificielle

## Vue d'ensemble

Cette première séance introduit les concepts fondamentaux de l'intelligence artificielle, son histoire, et ses applications actuelles. Nous explorons comment l'IA est définie par différents experts et organisations, et comment le domaine a évolué depuis ses débuts dans les années 1950.

---

## 1. Définitions de l'Intelligence Artificielle

### 1.1 Introduction

Après avoir exploré les définitions spontanées des étudiants via le questionnaire Wooclap, il est important de comprendre comment les experts et les organisations majeures du domaine définissent l'intelligence artificielle. Ces définitions ont évolué au fil du temps et reflètent différentes perspectives sur ce qu'est l'intelligence et comment elle peut être reproduite par des machines.

### 1.2 La définition fondatrice : Marvin Minsky (1950s)

**Définition :** "La science et l'ingénierie de la création de machines intelligentes"

```{admonition} Contexte historique
:class: note
Marvin Minsky est l'un des pères fondateurs du domaine de l'IA dans les années 1950. Cette définition est délibérément large et centrée sur l'objectif : créer des machines capables de fonctions cognitives.
```

**Concept clé :**
L'objectif principal est d'imiter les fonctions cognitives humaines telles que l'apprentissage et la résolution de problèmes ("to mimic human-like cognitive function such as learning and problem solving").

```{admonition} Point de discussion pour les étudiants
:class: tip
Cette définition soulève immédiatement la question : qu'est-ce que l'intelligence ? Pour créer une machine "intelligente", nous devons d'abord définir l'intelligence elle-même - un débat qui existe depuis longtemps en psychologie.
```

### 1.3 La définition d'IBM (perspective contemporaine)

**Définition :** "L'intelligence artificielle (IA) est une technologie qui permet aux ordinateurs et aux machines de simuler l'apprentissage, la compréhension, la résolution de problèmes, la prise de décision, la créativité et l'autonomie de l'être humain."

**Analyse comparative avec Minsky :**

La principale différence entre la définition d'IBM et celle de Minsky réside dans la précision du concept d'intelligence. IBM décompose l'intelligence en capacités spécifiques :

1. L'apprentissage
2. La compréhension
3. La résolution de problèmes
4. La prise de décision
5. La créativité
6. L'autonomie

```{admonition} Pourquoi cette évolution ?
:class: important
- Après 70 ans de recherche en IA, nous avons une meilleure compréhension des composantes de l'intelligence
- Cette définition est plus opérationnelle - elle permet de mesurer et d'évaluer les systèmes d'IA
- Elle reconnaît explicitement la "simulation" plutôt que la reproduction exacte
```

### 1.4 La définition de Coursera

**Définition :** "L'intelligence artificielle (IA) désigne les systèmes informatiques capables d'effectuer des tâches complexes que seul un être humain pouvait historiquement accomplir, telles que le raisonnement, la prise de décision ou la résolution de problèmes."

**Caractéristique distinctive :**
Cette définition met l'accent sur la relation avec l'intelligence humaine et sur l'aspect historique. Elle souligne que l'IA accomplit des tâches qui étaient auparavant l'apanage exclusif des humains.

```{admonition} Implication psychologique importante
:class: warning
Il s'agit essentiellement de définir l'intelligence par rapport à l'intelligence humaine comme référence. Cela pose la question : l'IA doit-elle nécessairement imiter l'humain, ou peut-elle développer sa propre forme d'intelligence ?
```

### 1.5 Synthèse des définitions

**Points communs à travers toutes les définitions :**

Si on analyse l'ensemble des définitions disponibles (Google, Britannica, Microsoft, etc.), plusieurs mots-clés reviennent constamment :

- Intelligence
- Humain/humaine
- Apprentissage
- Machines/ordinateurs
- Capacités/capabilities
- Résolution de problèmes
- Simulation
- Tâches
- Données

```{important}
**Conclusion fondamentale :**
La définition de l'IA dépend fondamentalement de votre définition de l'intelligence en général. C'est pourquoi le débat sur "qu'est-ce que l'intelligence ?" reste central, non seulement en psychologie mais aussi en IA.
```

---

## 2. Histoire et étapes clés de l'IA

### 2.1 Introduction à l'histoire de l'IA

Maintenant que nous avons une idée claire des définitions de l'IA, explorons comment tout cela a commencé et évolué jusqu'à aujourd'hui. L'histoire de l'IA n'est pas une progression linéaire - elle est marquée par des périodes d'enthousiasme intense suivies de désillusions ("hivers de l'IA"), puis de renaissances spectaculaires.

### 2.2 Les années 1950 : La naissance de l'IA

#### Le Test de Turing (1950)

**L'article fondateur :**
Alan Turing publie "Computing Machinery and Intelligence", proposant ce qui deviendra le célèbre Test de Turing.

```{admonition} Qu'est-ce que le Test de Turing ?
:class: note
Il est tout à fait possible que vous ayez déjà entendu parler du test de Turing. Si ce n'est pas le cas, il s'agit d'un test permettant de déterminer si un système est intelligent.
```

**Le principe du test :**

Le test est en fait une conversation entre le système et un humain. Si quelqu'un n'est pas capable de décider si son partenaire de conversation est un autre humain ou une machine, la machine passe le test.

En d'autres termes : **si quelqu'un n'est pas capable de distinguer l'humain de la machine dans une conversation, alors la machine est considérée comme intelligente.**

**Implications pour la psychologie :**
- Le test se base sur le comportement observable (approche béhavioriste)
- Il évite la question philosophique de la conscience ou de la "vraie" intelligence
- Il pose la question : l'imitation de l'intelligence équivaut-elle à l'intelligence ?

```{admonition} Pertinence aujourd'hui
:class: tip
Avec des systèmes comme ChatGPT, GPT-4, et Claude, nous sommes plus proches que jamais de "passer" le Test de Turing dans certains contextes. Mais cela signifie-t-il que ces systèmes sont vraiment intelligents ? C'est un débat toujours d'actualité.
```

#### La conférence de Dartmouth (1956)

**L'événement fondateur :**
La conférence de Dartmouth introduit officiellement le terme "Intelligence Artificielle". C'est la naissance officielle du domaine en tant que discipline scientifique distincte.

**Les participants clés incluaient :**
- John McCarthy (qui a inventé le terme "Intelligence Artificielle")
- Marvin Minsky
- Claude Shannon
- Allen Newell

#### Le Perceptron (1958)

**Innovation technique :**
Introduction du Perceptron, précurseur des réseaux neuronaux modernes.

```{important}
**Pourquoi est-ce important ?**
C'était la première tentative de créer un modèle mathématique inspiré du fonctionnement des neurones biologiques - un lien direct avec les neurosciences !
```

### 2.3 Les années 1960-1970 : Premiers succès et premier "hiver"

#### ELIZA (1964-1966) : Le premier chatbot

**Qu'est-ce qu'ELIZA ?**
ELIZA est la première chatbot qui pouvait converser en langage naturel. Elle a été créée par Joseph Weizenbaum au MIT.

**Le mode "psychothérapeute" :**

La version la plus célèbre d'ELIZA imitait un psychothérapeute rogérien, utilisant des techniques simples comme :
- Reformuler les déclarations du patient en questions
- Identifier des mots-clés et répondre avec des phrases pré-programmées
- Exemple : Patient : "Je suis triste" → ELIZA : "Pourquoi êtes-vous triste ?"

```{admonition} Découverte psychologique surprenante
:class: warning
Son inventeur, Weizenbaum, a été choqué de découvrir que les gens développaient des attachements émotionnels envers ELIZA et lui confiaient des informations personnelles profondes, **même en sachant qu'il s'agissait d'un programme informatique !**

C'était l'une des premières démonstrations de l'anthropomorphisation des machines.
```

**Pertinence pour les étudiants en psychologie :**
- Effet de l'illusion de compréhension
- Tendance humaine à attribuer de l'intelligence et de l'empathie aux machines
- Précurseur des débats actuels sur les chatbots thérapeutiques et l'IA en santé mentale

#### Les hivers de l'IA (années 1970)

```{admonition} Qu'est-ce qu'un "hiver de l'IA" ?
:class: note
Période durant laquelle le financement et l'intérêt pour la recherche en IA diminuent drastiquement.
```

**Causes du premier hiver :**
- Promesses non tenues : les chercheurs avaient prédit que l'IA générale serait réalisée dans 10-20 ans
- Limitations techniques : les ordinateurs n'étaient pas assez puissants
- Problèmes de complexité : les tâches s'avéraient beaucoup plus difficiles que prévu
- Critique du Perceptron (1969) par Minsky et Papert, montrant ses limitations

```{important}
**Leçon importante :**
Le développement de l'IA n'est pas linéaire. Les attentes irréalistes peuvent mener à des désillusions. C'est un cycle qui s'est répété plusieurs fois dans l'histoire de l'IA.
```

### 2.4 Les années 1980 : Renaissance avec les réseaux neuronaux

#### La rétropropagation (Backpropagation)

**Innovation majeure :**
Introduction de l'algorithme de rétropropagation pour entraîner les réseaux neuronaux multi-couches.

**Pourquoi est-ce révolutionnaire ?**
Avant cela, on ne savait pas comment entraîner efficacement des réseaux avec plusieurs couches de neurones. La rétropropagation permet au réseau d'"apprendre de ses erreurs" en ajustant les connexions de manière systématique.

```{admonition} Analogie avec l'apprentissage humain
:class: tip
C'est similaire à comment nous apprenons par essai-erreur et feedback. Le réseau fait une prédiction, compare avec la bonne réponse, et ajuste ses "synapses" (poids) en conséquence.
```

### 2.5 Les années 1990 : Passage au mainstream

#### Deep Blue vs. Garry Kasparov (1997)

**L'événement :**
Le jeu d'échecs entre Garry Kasparov (champion du monde) et Deep Blue d'IBM, dans lequel Deep Blue a battu Kasparov.

```{important}
**Pourquoi c'était si important ?**
- C'était la première fois qu'un ordinateur battait le champion du monde d'échecs
- Cela a montré au monde entier, de manière très visible et dramatique, le potentiel de l'IA
- Impact médiatique énorme - l'IA est devenue un sujet de conversation publique
```

**Ce que cela a révélé :**
Les échecs étaient longtemps considérés comme le summum de l'intelligence humaine. Cette victoire a forcé une réévaluation de ce que signifie "être intelligent".

**Changement de paradigme :**
Cette période a également marqué une transition vers les méthodes statistiques et le raisonnement probabiliste, s'éloignant des systèmes purement basés sur des règles logiques.

### 2.6 Les années 2010 : La révolution de l'apprentissage profond

#### AlexNet (2012) : Le tournant décisif

**L'innovation :**
AlexNet révolutionne la vision par ordinateur avec les réseaux neuronaux convolutifs (CNN - Convolutional Neural Networks).

```{important}
**Pourquoi AlexNet est-il si important ?**

L'invention d'AlexNet est sans doute à l'origine de tous les développements et progrès récents dans le domaine de l'IA. AlexNet est un réseau de neurones artificiels pour la reconnaissance d'objets à partir d'images, qui a largement dépassé tous les autres modèles d'IA de l'époque.
```

**Impact historique :**
Cette invention a redonné de l'espoir aux réseaux de neurones artificiels après des décennies de scepticisme. C'est le début de l'ère moderne de l'apprentissage profond (deep learning).

**Performance spectaculaire :**
Dans la compétition ImageNet 2012, AlexNet a réduit le taux d'erreur de 26% à 15% - une amélioration massive qui a stupéfié la communauté scientifique.

#### AlphaGo vs. Lee Sedol (2016)

**L'événement :**
AlphaGo (développé par DeepMind/Google) bat Lee Sedol, l'un des meilleurs joueurs de Go au monde.

**Pourquoi le Go est différent des échecs :**
- Le Go a environ 10^170 positions possibles (vs. 10^50 pour les échecs)
- Impossible d'utiliser la force brute pour explorer toutes les possibilités
- Nécessite de l'intuition et du "jugement" plutôt que du calcul pur

**L'innovation technique :**
AlphaGo utilise l'apprentissage par renforcement profond, combinant :
- Réseaux neuronaux pour évaluer les positions
- Apprentissage par auto-jeu (la machine joue contre elle-même)
- Monte Carlo Tree Search pour la planification

#### GPT-2 (2018) et les débuts de l'IA générative

**Innovation :**
GPT-2 montre des capacités avancées de modélisation du langage, capable de générer du texte cohérent et contextuellement approprié.

**Controverse :**
OpenAI a initialement refusé de publier le modèle complet par crainte d'abus (désinformation, fake news). C'était l'un des premiers débats publics sur la responsabilité en IA.

### 2.7 Les années 2020 : L'IA omniprésente

**Développements récents :**
- ChatGPT (2022) : explosion de l'utilisation grand public
- GPT-4, Claude, Gemini : modèles de langage de plus en plus sophistiqués
- DALL-E, Midjourney, Stable Diffusion : génération d'images
- Sora : génération de vidéos

**L'IA est désormais partout :**
Des assistants vocaux aux recommandations Netflix, de la détection de spam à la traduction automatique, l'IA fait maintenant partie de notre quotidien.

### 2.8 Les "Godfathers of AI" et le lien avec Montréal

```{important}
L'IA que nous avons aujourd'hui est le résultat des efforts de personnes qui croyaient en la puissance des réseaux neuronaux artificiels pendant les années où peu de chercheurs y croyaient.
```

**Les trois pionniers :**

Les recherches de trois chercheurs en particulier ont été déterminantes pour l'avancement de l'IA et de l'apprentissage profond :

1. **Geoffrey Hinton** (Université de Toronto)
2. **Yann LeCun** (maintenant chez Meta/Facebook)
3. **Yoshua Bengio** (Université de Montréal)

**Ces trois chercheurs sont connus comme les "Godfathers of AI" (les parrains de l'IA).**

```{admonition} Le lien avec l'UdeM et Montréal
:class: note
Yoshua Bengio est professeur au département d'informatique et de recherche opérationnelle (DIRO) de l'Université de Montréal.

Grâce à ses efforts depuis plus de 30 ans, Montréal est maintenant mondialement reconnue pour la recherche en IA. L'écosystème de l'IA que nous avons aujourd'hui à Montréal est principalement dû aux efforts du professeur Bengio et de ses collaborateurs.
```

**Institutions importantes à Montréal :**
- [MILA (Montreal Institute for Learning Algorithms)](https://mila.quebec/)
- [IVADO (Institute for Data Valorization)](https://ivado.ca/)
- Nombreuses startups d'IA
- Bureaux de recherche de grandes entreprises (Google, Meta, Microsoft, etc.)

**Prix Nobel de physique 2024 :**
Geoffrey Hinton et John Hopfield ont reçu le prix Nobel de physique en 2024 pour leurs travaux fondateurs sur les réseaux neuronaux artificiels - reconnaissance ultime de l'importance de ce domaine.

---

## 3. Exemples et applications de l'IA

### 3.1 Vision par ordinateur : Détection et reconnaissance de visages

**Technologie :**
Les systèmes modernes peuvent détecter des visages même dans des images très complexes avec de nombreuses personnes.

**Comment ça fonctionne (niveau conceptuel) :**
- Le modèle analyse l'image pixel par pixel
- Il recherche des patterns caractéristiques des visages (yeux, nez, bouche, proportions)
- Il attribue un score de confiance (probabilité)

```{admonition} Exemple
:class: tip
Quand vous voyez un chiffre comme "0.98" à côté d'un visage détecté, cela indique que le modèle est très sûr (98% de confiance) qu'il y a un visage à cet endroit.
```

**Limites et erreurs :**
Même les meilleurs modèles font des erreurs. Par exemple, un système pourrait incorrectement identifier deux visages différents comme étant la même personne, ou manquer un visage dans certaines conditions (éclairage, angle, occlusion).

**Applications :**
- Déverrouillage de téléphones
- Identification aux frontières
- Marquage automatique sur les réseaux sociaux
- Surveillance (avec implications éthiques importantes)

```{admonition} Questions éthiques pour discussion
:class: warning
- Vie privée et surveillance
- Biais raciaux dans les systèmes de reconnaissance faciale
- Consentement et utilisation des données biométriques
```

**Démo interactive :**
[Face detection demo](https://justadudewhohacks.github.io/face-api.js/face_and_landmark_detection)

### 3.2 Reconnaissance des émotions

**Objectif :**
Les systèmes tentent d'identifier l'état émotionnel d'une personne à partir de son expression faciale.

**Pertinence pour la psychologie :**
- Basé sur la théorie des émotions universelles de Paul Ekman
- Mais : débat sur la fiabilité de la reconnaissance d'émotions par l'expression faciale seule
- Contexte culturel et individuel

**Applications potentielles :**
- Détection de l'engagement en éducation
- Analyse du comportement des consommateurs
- Assistance aux personnes avec autisme

```{admonition} Controverses
:class: warning
De nombreux psychologues critiquent ces systèmes, arguant que :
- Les émotions sont complexes et ne se réduisent pas aux expressions faciales
- Le contexte est crucial
- Risque de sur-simplification de l'expérience émotionnelle humaine
```

### 3.3 IA générative : Text-to-Image

**Exemples de systèmes :**
- [DALL-E (OpenAI)](https://openai.com/index/dall-e-3/)
- [Midjourney](https://www.midjourney.com/home)
- Stable Diffusion

```{admonition} Exemple célèbre
:class: note
**Prompt:** "An illustration of an avocado sitting in a therapist's chair, saying 'I just feel so empty inside' with a pit-sized hole in its center. The therapist, a spoon, scribbles notes."

Ce prompt généré par DALL-E est devenu viral, démontrant la capacité créative et humoristique de l'IA.
```

**Comment ça fonctionne (niveau conceptuel) :**
1. Le système a appris les relations entre des millions de paires image-texte
2. Il comprend les concepts (avocat, thérapeute, vide, etc.)
3. Il peut combiner ces concepts de manière créative et cohérente

**Implications :**
- Démocratisation de la création artistique
- Questions sur l'originalité et les droits d'auteur
- Potentiel pour l'illustration, le design, la publicité

### 3.4 IA générative : Text-to-Video

**Exemple principal :** [Sora (OpenAI)](https://openai.com/index/sora/)

**Capacités :**
Génération de vidéos réalistes à partir de descriptions textuelles.

```{admonition} Implications et risques
:class: danger
Un des risques majeurs est la capacité de **"leurer" ou "tromper"** les gens avec des vidéos fausses mais réalistes (deepfakes).

**Deepfakes - Un problème sérieux :**
- Vidéos manipulées de personnalités publiques
- Potentiel de désinformation à grande échelle
- Utilisation malveillante (revenge porn, fraude, manipulation politique)
- Difficulté croissante à distinguer le vrai du faux
```

**Questions éthiques urgentes :**
- Comment protéger la vérité et l'authenticité ?
- Responsabilité des créateurs de ces technologies
- Nécessité de littératie numérique
- Régulation et législation

### 3.5 Applications quotidiennes de l'IA

**Google Photos :**
Tous ceux qui utilisent Google Photo savent qu'il doit y avoir une intelligence artificielle derrière. Le système peut :
- Reconnaître et regrouper automatiquement les visages
- Identifier des objets, animaux, lieux
- Créer des albums thématiques automatiquement
- Rechercher par contenu ("montre-moi les photos de plage")

**Autres exemples quotidiens :**
- Recommandations Netflix, Spotify, YouTube
- Assistant vocaux (Siri, Alexa, Google Assistant)
- Correction automatique et prédiction de texte
- Traduction automatique (Google Translate, DeepL)
- Filtres anti-spam dans les emails
- Navigation GPS et prédiction de trafic
- Détection de fraude bancaire

```{admonition} Question pour réflexion
:class: tip
Quoi d'autre ? Pensez à d'autres exemples d'IA que vous utilisez quotidiennement !
```

---

## 4. Points de réflexion pour les étudiants en psychologie

### 4.1 L'IA comme outil de recherche en psychologie

**Opportunités :**
- Analyse de grandes quantités de données comportementales
- Modélisation des processus cognitifs
- Détection de patterns invisibles à l'œil humain
- Personnalisation des interventions thérapeutiques

### 4.2 L'IA comme objet d'étude psychologique

**Questions de recherche :**
- Comment les humains interagissent-ils avec l'IA ?
- Anthropomorphisation des machines
- Confiance et acceptation de l'IA
- Impact sur l'emploi et l'identité professionnelle
- Effets psychologiques de l'automatisation

### 4.3 Liens avec les neurosciences cognitives

**Inspiration mutuelle :**
- Les réseaux neuronaux artificiels sont inspirés du cerveau
- L'IA peut aider à comprendre le fonctionnement cérébral
- Approche computationnelle des neurosciences
- Modélisation des processus cognitifs

### 4.4 Questions éthiques

**Thèmes importants :**
- Biais algorithmiques et discrimination
- Vie privée et surveillance
- Autonomie et prise de décision
- Responsabilité et transparence
- Impact social et inégalités

---

## 5. Concepts clés à retenir

```{important}
1. **La définition de l'IA** dépend fondamentalement de comment on définit l'intelligence
2. **L'histoire de l'IA** est cyclique : enthousiasme → désillusion → renaissance
3. **Les années 2010** marquent un tournant avec l'apprentissage profond (deep learning)
4. **Montréal** est un centre mondial de recherche en IA grâce à Yoshua Bengio
5. **L'IA moderne** est omniprésente dans notre quotidien
6. **Les applications** vont de la vision par ordinateur à la génération de contenu
7. **Les enjeux éthiques** sont cruciaux et multidimensionnels
```

---

## 6. Ressources complémentaires

### Podcast recommandé

```{admonition} À écouter
:class: tip
**Radio-Canada OHdio - "Fascinant"**
Épisode sur Deep Blue vs. Kasparov

[Écouter le podcast](https://ici.radio-canada.ca/ohdio/balados/8225/fascinant/521128/intelligence-artificielle-ordinateur-deep-blue-echecs-garry-kasparov-robot-machines)
```

### Sites web pour explorer

- [Face detection demo](https://justadudewhohacks.github.io/face-api.js/face_and_landmark_detection)
- [DALL-E](https://openai.com/index/dall-e-3/)
- [Midjourney](https://www.midjourney.com/home)
- [Sora](https://openai.com/index/sora/)

### Institutions à Montréal

- [MILA](https://mila.quebec/)
- [IVADO](https://ivado.ca/)

---

## 7. Questions pour la discussion en classe

```{admonition} Questions de réflexion
:class: question
1. Comment votre définition personnelle de l'IA a-t-elle évolué après ce cours ?
2. Le Test de Turing est-il encore pertinent aujourd'hui ? Pourquoi ou pourquoi pas ?
3. Quelles sont les implications psychologiques des "hivers de l'IA" pour les chercheurs ?
4. Comment l'expérience ELIZA éclaire-t-elle notre compréhension de l'anthropomorphisation ?
5. Quels sont les risques psychologiques et sociaux des deepfakes ?
6. Comment l'IA pourrait-elle transformer la pratique de la psychologie clinique ?
7. Quelles compétences psychologiques sont difficiles ou impossibles à automatiser ?
```

---

## Notes pour l'enseignant

```{admonition} Timing suggéré
:class: note
- Définitions : 15-20 minutes
- Histoire (avec Test de Turing détaillé) : 25-30 minutes
- Exemples et applications : 15-20 minutes
- Discussion : 10-15 minutes
```

**Activités interactives :**
- Questionnaire Wooclap avant et après les définitions
- Discussion sur le Test de Turing
- Démonstrations des applications (face detection, génération d'images)
- Débat sur les deepfakes

**Points d'attention :**
- Relier constamment les concepts d'IA aux connaissances psychologiques des étudiants
- Encourager la pensée critique sur les capacités et limitations de l'IA
- Aborder les questions éthiques de manière nuancée
- Montrer l'enthousiasme pour le domaine tout en restant réaliste sur les défis
