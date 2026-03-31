# TODO

## Pour une classification binaire avec sorties booléennes, voici ce qui manque encore :

  ---
  1. Analyse des erreurs par catégorie

  Plutôt que des métriques globales, décomposer les FP/FN par dossier source (false_doc, false_id, jdd, id_cards, passports) pour identifier où chaque modèle échoue. Les false_id semblent être la source
  principale des FP — le confirmer quantitativement.

  2. Matrice de confusion sur id_doc_type (multi-classe)

  is_doc_id est binaire mais id_doc_type a 4 valeurs (id card, passport, proof_of_residency, not_identity_doc). Une confusion matrix multi-classe révélerait les confusions entre types (ex: passeport
  classifié en id card). Métriques adaptées : F1 macro/weighted, Cohen's Kappa.

  3. Accord inter-modèles (Cohen's Kappa)

  Mesurer dans quelle mesure les modèles s'accordent entre eux indépendamment de la vérité terrain. Un Kappa élevé entre deux modèles qui se trompent pareillement révèle une faiblesse systémique (ex: tous
  confondent la même catégorie false_id).

  4. Analyse de robustesse par type de fichier

  Comparer les performances sur .jpg vs .png vs .pdf/.webp — certains modèles peuvent être plus fragiles sur des formats ou des qualités d'image spécifiques.

  5. Temps de réponse par modèle

  En production la latence compte. Chronométrer chaque appel (time.perf_counter autour de runner()) et comparer le ratio performance/latence pour choisir le meilleur compromis.

  ---
  Priorité suggérée : commencer par 1 (immédiatement exploitable avec les données actuelles) puis 2 (exploite id_doc_type déjà disponible dans results.csv).

## Catégorisation multi-label / multi-classe en Data Science

A ce stade on ne fait de la mesure de perf que sur la feature 'is_id_doc', on devrait faire une mesure de perf sur chaque label.

---

## 1. Distinguer les cas d'usage

| Type | Description | Exemple |
|---|---|---|
| **Multi-classe** | Une seule étiquette parmi N classes | Sentiment : positif / négatif / neutre |
| **Multi-label** | Plusieurs étiquettes simultanées possibles | Un document → [finance, risque, ESG] |
| **Hiérarchique** | Classes organisées en arborescence | Catégorie → sous-catégorie → type |

---

## 2. Principes clés

**Passage d'une feature à plusieurs**
- Chaque feature supplémentaire apporte de l'information discriminante, mais aussi du bruit potentiel → nécessité de **sélection de features**
- Attention à la **corrélation entre features** (multicolinéarité) et à la **malédiction de la dimensionnalité**
- Les features peuvent être de natures différentes (numériques, catégorielles, textuelles, embeddings) → preprocessing adapté par type

**Séparation des classes**
- Avec une seule feature, la frontière est un seuil 1D → avec N features, c'est un **hyperplan** (ou surface non linéaire)
- Plus les classes sont **linéairement séparables** dans l'espace des features, plus les modèles simples suffisent

---

## 3. Méthodes principales

### Approches classiques (interprétables)
- **Logistic Regression** (one-vs-rest ou multinomial) — bon baseline
- **Decision Tree / Random Forest** — nativement multi-classe, feature importance intégrée
- **SVM** (one-vs-one ou one-vs-rest) — efficace sur espaces de haute dimension
- **Naive Bayes** — rapide, bien adapté au texte

### Approches ensemblistes
- **Gradient Boosting** (XGBoost, LightGBM, CatBoost) — souvent le meilleur compromis performance/explicabilité en produit
- **Voting / Stacking** — combine plusieurs modèles

### Approches deep learning / NLP
- **Embeddings + couche dense** (ex. sentence-transformers + classification head)
- **Fine-tuning LLM** (CamemBERT, etc.) si le texte est la feature principale
- **Few-shot / zero-shot** avec LLM pour des catégorisations sans données étiquetées

### Stratégies multi-label spécifiques
- **Binary Relevance** : un classifieur binaire par label (simple mais ignore les corrélations)
- **Classifier Chains** : chaque classifieur prend en entrée les prédictions des précédents
- **Label Powerset** : traite chaque combinaison de labels comme une classe unique (explose si beaucoup de labels)

---

## 4. Métriques

### Multi-classe (une classe par exemple)

| Métrique | Ce qu'elle mesure | Quand l'utiliser |
|---|---|---|
| **Accuracy** | % de bonnes prédictions globales | Classes équilibrées uniquement |
| **Precision / Recall / F1** par classe | Performance sur chaque classe | Toujours utile |
| **Macro F1** | Moyenne non pondérée des F1 par classe | Quand chaque classe compte également |
| **Weighted F1** | Moyenne pondérée par support | Quand les classes sont déséquilibrées |
| **Confusion Matrix** | Erreurs inter-classes | Debug et analyse d'erreurs |
| **Cohen's Kappa** | Accord au-delà du hasard | Utile en NLP/annotation |

### Multi-label (plusieurs labels par exemple)

| Métrique | Description |
|---|---|
| **Hamming Loss** | % de labels mal prédits sur l'ensemble des paires (exemple, label) |
| **Subset Accuracy** | Exactitude stricte : tous les labels doivent être corrects |
| **Micro F1** | Agrège TP/FP/FN sur tous les labels — favorise les labels fréquents |
| **Macro F1** | Moyenne des F1 par label — équitable entre labels rares et fréquents |
| **Sample F1** | F1 calculé par exemple puis moyenné |

---

## 5. Pièges fréquents

- **Déséquilibre de classes** → oversampling (SMOTE), undersampling, class weights, ou seuils de décision différents par classe
- **Fuite de données** → le preprocessing (normalisation, encodage) doit être fité uniquement sur le train set
- **Métrique mal choisie** → l'accuracy sur des classes déséquilibrées est trompeuse, toujours regarder la confusion matrix
- **Threshold par défaut à 0.5** en multi-label → optimiser le seuil par classe selon le contexte métier (précision vs rappel)

