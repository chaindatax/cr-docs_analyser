
## Solution Proposée

Overview synaptique de la solution, intégrée avec les applications et les services existants (GED, Systèmes legacy, opérationnels, ...)


![img.png](docs/diagram_overview.png)

Overview de la solution, l'application 'data science' créée dans ce repo.

![data_science_project.png](docs/data_science_project.png)


## Résultats de l'analyse & performances des modèles

![docs/img.png](docs/img.png)

## Conclusion — Performance des modèles sur `is_doc_id`

L'analyse porte sur **40 fichiers** labellisés (22 documents d'identité, 18 non-identité), évalués sur trois modèles disposant de résultats complets. Azure CU n'a produit aucune prédiction et est exclu.

> **Note méthodologique** — ROC AUC n'est pas utilisé ici : il requiert des scores de confiance continus pour faire varier un seuil. Avec des sorties booléennes, la "courbe" ROC se réduit à un unique point et l'AUC est équivalente à la balanced accuracy, sans apport supplémentaire. Les matrices de confusion et les métriques F1/MCC sont plus adaptées.

### Résultats

| Modèle | Accuracy | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|
| **Mistral OCR** | **0.78** | **0.71** | **1.00** | **0.83** | **0.60** |
| Mistral Vision | 0.75 | 0.69 | 1.00 | 0.81 | 0.55 |
| Azure Vision | 0.75 | 0.69 | 1.00 | 0.81 | 0.55 |

### Points clés

**Recall parfait (1.00) sur les trois modèles** — aucun document d'identité réel n'est manqué. C'est la propriété la plus critique en vérification documentaire : un faux négatif représente un risque métier majeur.

**Mistral OCR est le meilleur modèle** (F1 = 0.83, MCC = 0.60), devant Mistral Vision et Azure Vision à égalité (F1 = 0.81, MCC = 0.55). L'écart est faible, ce qui suggère que l'OCR n'apporte pas d'avantage décisif sur la vision pure pour cette tâche.

**Precision modérée (~0.69–0.71)** — 9 à 10 faux positifs sur 40 fichiers, tous issus de la catégorie `false_id` (photos de documents sur Internet). Cette catégorie est intrinsèquement ambiguë pour les modèles.

**MCC entre 0.55 et 0.60** — performance correcte mais en deçà du seuil recommandé pour la production (> 0.80).

### Limites

- **Dataset petit** (n=40) — les métriques sont sensibles à 1 ou 2 exemples.
- **Azure CU non évalué** — à relancer.
- **`false_id` ambiguë** — images de documents web, pas des faux documents réels.

### Recommandations

1. **Étendre le dataset** avec plus de cas négatifs réels pour réduire les faux positifs.
2. **Relancer Azure CU** pour la comparaison complète sur 4 modèles.
3. **Exploiter `id_doc_type`** comme filtre secondaire pour affiner la décision au-delà du booléen.


## Métriques d'évaluation — rappel

| Métrique | Formule | Ce qu'elle mesure | Plage | Objectif | Quand l'utiliser |
|---|---|---|---|---|---|
| **Accuracy** | (TP + TN) / Total | Part des prédictions correctes (toutes classes) | [0, 1] | → 1 | Classes équilibrées |
| **Precision** | TP / (TP + FP) | Parmi les positifs prédits, combien sont vrais | [0, 1] | → 1 | Coût élevé des faux positifs |
| **Recall** | TP / (TP + FN) | Parmi les vrais positifs, combien sont détectés | [0, 1] | → 1 | Coût élevé des faux négatifs |
| **F1** | 2 × (P × R) / (P + R) | Moyenne harmonique Precision/Recall | [0, 1] | → 1 | Classes déséquilibrées |
| **MCC** | (TP·TN − FP·FN) / √(...) | Corrélation entre prédictions et réalité | [−1, 1] | → 1 | Déséquilibre sévère, vision globale |

> **Légende :** TP = vrai positif · TN = vrai négatif · FP = faux positif · FN = faux négatif  
> **Valeurs de référence MCC :** −1 = prédictions systématiquement inverses · 0 = aléatoire · +1 = parfait

### Règles d'or
- **Accuracy** trompe sur des classes déséquilibrées (ex : 95% de classe majoritaire).  
- **Precision vs Recall** : arbitrage selon le coût métier de chaque type d'erreur.  
- **F1** synthétise les deux, mais suppose que Precision et Recall ont le même poids.  
- **MCC** est le seul indicateur symétrique sur toutes les cases de la matrice de confusion — privilégié en contexte médical, fraud detection, KYC/AML.
- 

