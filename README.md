# Docs Analyser

Batch analysis of identity documents (ID cards, passports) using four independent AI backends — Mistral OCR, Mistral Vision (Pixtral), Azure Content Understanding, and Azure OpenAI Vision — with result comparison.


## Solution Proposée

Overview synaptique de la solution, intégrée avec les applications et les services existants (GED, Systèmes legacy, opérationnels, ...)


![img.png](docs/diagram_overview.png)

Overview de la solution, l'application 'data science' créée dans ce repo.

![data_science_project.png](docs/data_science_project.png)


## Résultats de l'analyse & performances des modèles

![docs/img.png](docs/img.png)

### Conclusion — Performance des modèles sur `is_doc_id`

L'analyse porte sur **40 fichiers** labellisés (22 documents d'identité, 18 non-identité), évalués sur les **quatre modèles** — tous ont produit des prédictions complètes, sauf Mistral Vision (n=39, une prédiction manquante).

### Résultats

| Modèle | n | Accuracy | Precision | Recall | F1 | MCC | TP | TN | FP | FN |
|---|---|---|---|---|---|---|---|---|---|---|
| **Azure CU** | **40** | **0.80** | **0.73** | **1.00** | **0.85** | **0.64** | 22 | 10 | 8 | 0 |
| Mistral OCR | 40 | 0.78 | 0.71 | 1.00 | 0.83 | 0.60 | 22 | 9 | 9 | 0 |
| Azure Vision | 40 | 0.78 | 0.71 | 1.00 | 0.83 | 0.60 | 22 | 9 | 9 | 0 |
| Mistral Vision | 39 | 0.77 | 0.71 | 1.00 | 0.83 | 0.58 | 22 | 8 | 9 | 0 |

### Points clés

**Recall parfait (1.00) sur les quatre modèles** — aucun document d'identité réel n'est manqué. C'est la propriété la plus critique en vérification documentaire : un faux négatif représente un risque métier majeur.

**Azure CU est le meilleur modèle** (F1 = 0.85, MCC = 0.64, 8 FP), devançant Mistral OCR et Azure Vision à égalité (F1 = 0.83, MCC = 0.60, 9 FP). Mistral Vision est légèrement en retrait (MCC = 0.58) et compte une prédiction manquante.

**Tous les faux positifs proviennent du dossier `false_id`** — images de documents d'identité trouvées sur Internet. Ces visuels ressemblent à de vrais documents (résolution acceptable, structure visible) : le modèle réagit à l'apparence du document, pas à son authenticité. C'est un biais structurel du jeu de données, pas une erreur de modèle.

**MCC entre 0.58 et 0.64** — performance correcte mais en deçà du seuil recommandé pour la production (> 0.80). L'écart entre les modèles est faible : l'approche OCR n'apporte pas d'avantage décisif sur la vision pure pour cette tâche binaire.

### Limites

- **Dataset petit** (n=40) — les métriques sont sensibles à 1 ou 2 exemples (1 FP de moins = +0.04 MCC).
- **`false_id` ambiguë** — images de documents web, pas des faux documents réels. Les FP sur cette catégorie ne reflètent pas forcément le comportement en production.
- **3 fichiers non analysés** — les documents CORUM et l'attestation PDF ne figurent pas dans `results.csv`.

### Recommandations

1. **Azure CU à privilégier** pour les déploiements, avec 1 FP de moins que les autres modèles.
2. **Étendre le dataset** avec des cas négatifs réels (factures, relevés bancaires) pour confirmer la robustesse — les 9 faux positifs actuels sont tous des images de documents.
3. **Exploiter `id_doc_type`** comme filtre secondaire (passeport vs carte d'identité) pour affiner la décision au-delà du booléen.


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

## Setup

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync
```

Create a `.env` file at the project root:

```
MISTRAL_API_KEY=your_mistral_key

CONTENTUNDERSTANDING_ENDPOINT=https://your-resource.services.ai.azure.com/
CONTENTUNDERSTANDING_KEY=your_azure_key

AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_KEY=your_azure_openai_key
```

### Azure prerequisites

Both Azure analysers require models deployed in your Azure AI Foundry resource:

- `gpt-4.1` — field extraction (Content Understanding) and vision classification (Azure OpenAI)
- `text-embedding-3-large` — semantic document chunking (Content Understanding only)

## Usage

```bash
uv run main.py        # analyse all files in dataset/ and write results.csv
uv run pytest tests/  # run unit tests
uv run pytest tests/test_dataset.py -v  # run integration tests against real APIs
```

Place images (`.jpg` or `.png`) in `dataset/` before running. The output `results.csv` is written to the project root.

## Architecture

```
docs_analyser/
├── base.py                    # Analyser ABC + AnalysisResult dataclass
├── mistral_analyser.py        # MistralAnalyser       — Mistral OCR API
├── mistral_vision_analyser.py # MistralVisionAnalyser — Pixtral vision model
├── azure_analyser.py          # AzureAnalyser         — Azure Content Understanding
└── azure_vision_analyser.py   # AzureVisionAnalyser   — Azure OpenAI vision model
main.py                        # async batch runner, writes results.csv
tests/
├── test_main.py               # unit tests (mocked)
└── test_dataset.py            # integration tests (real API calls, skipped if keys absent)
```

### Base classes (`base.py`)

`Analyser` is an abstract base class with a single method:

```python
def runner(self, file_path: str) -> AnalysisResult
```

`AnalysisResult` is a dataclass with three fields:

| Field | Type | Description |
|---|---|---|
| `id_doc` | `bool` | Whether the document is an identity document |
| `document_id_type` | `str` | `"id card"`, `"passport"`, `"proof_of_residency"`, or `"not_identity_doc"` |
| `document_type` | `str` | Free-form document type as described by the model |

### MistralAnalyser

Encodes the image as base64 and calls the `mistral-ocr-latest` model with a structured JSON schema to extract all three fields from the document text content.

### MistralVisionAnalyser

Sends the image to `pixtral-12b-2409` via the chat completion API with a prompt requesting JSON output. Classifies the document visually without relying on OCR text extraction — works even when text is blurry or partially obscured.

### AzureAnalyser

Uses Azure Content Understanding with a custom analyzer (`identityDocClassifier`) built on top of `prebuilt-document`. The analyzer is created automatically on first run and reused on subsequent calls. Uses `gpt-4.1` for field extraction and `text-embedding-3-large` for document chunking.

### AzureVisionAnalyser

Sends the image to `gpt-4.1` (vision-capable) via Azure OpenAI chat completions with a prompt requesting JSON output. Like the Mistral vision analyser, classifies the document visually without OCR.

### Batch runner (`main.py`)

Processes all images in `dataset/` in batches of 10 using `asyncio`:

- Each batch runs files concurrently via `asyncio.gather`
- Within each file, all four analyser calls run in parallel via `asyncio.gather` + `asyncio.to_thread`
- Results are written to `results.csv` preserving the original file order

### Output CSV

| Column | Description |
|---|---|
| `file_path` | Relative path to the image |
| `filename` | Image filename |
| `mistral_id_doc` | `id_doc` from Mistral OCR |
| `mistral_document_id_type` | `document_id_type` from Mistral OCR |
| `mistral_document_type` | `document_type` from Mistral OCR |
| `azure_id_doc` | `id_doc` from Azure Content Understanding |
| `azure_document_id_type` | `document_id_type` from Azure Content Understanding |
| `azure_document_type` | `document_type` from Azure Content Understanding |
| `mistral_vision_id_doc` | `id_doc` from Mistral Vision |
| `mistral_vision_document_id_type` | `document_id_type` from Mistral Vision |
| `mistral_vision_document_type` | `document_type` from Mistral Vision |
| `azure_vision_id_doc` | `id_doc` from Azure OpenAI Vision |
| `azure_vision_document_id_type` | `document_id_type` from Azure OpenAI Vision |
| `azure_vision_document_type` | `document_type` from Azure OpenAI Vision |
| `aligned` | `True` if all four analysers agree on all three fields |