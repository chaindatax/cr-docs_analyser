


![img.png](docs/img.png)

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



