Pour une classification binaire avec sorties booléennes, voici ce qui manque encore :

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

