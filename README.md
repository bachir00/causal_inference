# Projets d'Inférence Causale et A/B Testing

Ce dépôt contient deux notebooks dédiés à l'analyse causale et aux tests A/B, utilisant des méthodes avancées de Machine Learning causal.

## 📊 Vue d'ensemble

### 1. **causal_lalonde_notebook.ipynb** - Analyse causale du dataset LaLonde
### 2. **notebook.ipynb** - A/B Testing avec Causal ML

---

## 📓 Notebook 1 : Analyse Causale LaLonde

**Objectif :** Estimer l'impact causal d'un programme de formation sur le revenu des participants en 1978.

### Dataset
- **Source :** Dataset LaLonde (Matching::lalonde)
- **Traitement :** Programme de formation (`treat`)
- **Outcome :** Revenu en 1978 (`re78`)
- **Covariables :** age, éducation, race (black, hispan), statut marital, diplôme, revenus 1974-1975

### Méthodologie

#### 1. **Exploration des données (EDA)**
- Statistiques descriptives complètes
- Analyse de la distribution du revenu par groupe (traité vs contrôle)
- Vérification des valeurs manquantes
- Analyse de la proportion traité/contrôle

#### 2. **Modélisation causale par DAG**
Construction d'un graphe acyclique dirigé (DAG) pour:
- Identifier les confounders (variables confondantes)
- Visualiser les relations causales entre variables
- Documenter les hypothèses causales
- Deux versions : DAG détaillé et DAG simplifié

#### 3. **Estimation du Propensity Score**
- Modèle : Régression logistique
- Normalisation des covariables (StandardScaler)
- Calcul de `P(treat=1 | covariates)` pour chaque individu
- Objectif : Équilibrer les groupes comme dans une expérimentation randomisée

#### 4. **Vérification de l'overlap**
- Visualisation des densités de Propensity Score
- Comparaison traités vs contrôles
- Analyse des moyennes par groupe (PS moyen contrôle: 0.18, traité: 0.57)

#### 5. **Estimation de l'ATE (Average Treatment Effect)**

##### a) **Matching sur Propensity Score**
- Algorithme : Nearest Neighbor (1-NN)
- Chaque individu traité est apparié au contrôle le plus proche
- **ATE ≈ 1227 $**

##### b) **Bootstrap pour intervalle de confiance**
- 2000 itérations
- Intervalle de confiance à 95%
- Validation de la robustesse de l'estimation

##### c) **IPW (Inverse Probability Weighting)**
- Calcul des poids stabilisés
- Estimation alternative de l'ATE
- Méthode complémentaire au matching

#### 6. **Diagnostics de qualité**

##### Standardized Mean Difference (SMD)
- Indicateur clé de l'équilibre des covariables
- Comparaison avant/après matching
- Objectif : SMD < 0.1 pour un bon équilibre
- Vérification pour toutes les covariables

#### 7. **Causal Forest pour CATE**
- **Outil :** EconML CausalForestDML
- **Modèles :** RandomForest pour outcome et traitement
- Estimation de l'ATE global
- **Calcul des CATE** (Conditional Average Treatment Effect) individuels

##### Analyses CATE réalisées :
- Distribution des effets individuels (histogrammes)
- Segmentation par effet (négatif / quasi-nul / positif)
- Analyse par sous-groupes (ex: effet selon `nodegree`)
- Identification des individus avec fort effet positif (CATE > 500)
- Visualisations avec zones colorées

#### 8. **Interprétation des résultats**
- Les CATE négatifs indiquent que certains individus auraient un revenu inférieur avec le traitement
- Distribution hétérogène des effets → importance de la personnalisation
- Pourcentages d'individus par catégorie d'effet

### Technologies utilisées
```python
pandas, numpy, statsmodels
sklearn (LogisticRegression, NearestNeighbors, StandardScaler)
matplotlib, seaborn, networkx
econml (CausalForestDML)
```

---

## 📓 Notebook 2 : A/B Test Causal ML

**Objectif :** Simuler et analyser un test A/B pour mesurer l'impact d'une campagne publicitaire sur les achats clients.

### Dataset simulé
- **Taille :** 1000 clients
- **Variables :** age, income, historical_purchase
- **Traitement :** Exposition à la publicité (treatment = 0 ou 1)
- **Outcome :** Montant d'achat (purchase)
- **Effet simulé :** +100$ pour le groupe traité + bruit aléatoire

### Méthodologie

#### 1. **Simulation de données**
- Génération aléatoire avec seed fixe (reproductibilité)
- Assignment aléatoire du traitement (p=0.5)
- Simulation d'un effet causal connu (+100$)

#### 2. **Vérification de la randomisation**
- Comparaison des moyennes par groupe
- Validation que l'assignment est bien aléatoire
- Équilibre des covariables entre traités et contrôles

#### 3. **Estimation du Propensity Score**
- Régression logistique sur les covariables
- Même si randomisé, utile pour démonstration pédagogique
- Calcul de `P(treatment=1 | age, income, historical_purchase)`

#### 4. **Vérification de l'overlap**
- Densités des Propensity Scores
- Visualisation par groupe (traité vs contrôle)
- Validation de la zone de support commun

#### 5. **Estimation ATE**

##### a) **Matching simple**
- Nearest Neighbor sur Propensity Score
- Chaque traité apparié à son plus proche voisin contrôle
- Calcul de l'ATE moyen

##### b) **IPW stabilisé**
- Calcul des poids inversement proportionnels au PS
- Stabilisation pour réduire la variance
- Estimation pondérée de l'ATE

#### 6. **Causal Forest pour CATE**
- **Modèle :** CausalForestDML (EconML)
- Estimation de l'ATE global
- **Calcul des CATE individuels** pour personnalisation

##### Analyses réalisées :
- Distribution des effets individuels
- Histogrammes des CATEs
- Identification des clients avec effet positif (CATE > 90)

### Résultats attendus
Étant donné que l'effet est simulé à +100$, les estimations ATE devraient être proches de cette valeur, validant ainsi les méthodes.

### Technologies utilisées
```python
pandas, numpy
sklearn (LogisticRegression, NearestNeighbors, RandomForestRegressor)
matplotlib, seaborn
econml (CausalForestDML)
```

---

## 🔑 Concepts clés utilisés

### Propensity Score (PS)
Probabilité de recevoir le traitement conditionnellement aux covariables. Permet de réduire le biais de sélection.

### ATE (Average Treatment Effect)
Effet moyen du traitement sur l'ensemble de la population.

### CATE (Conditional Average Treatment Effect)
Effet du traitement pour un individu ou sous-groupe spécifique. Permet la personnalisation.

### SMD (Standardized Mean Difference)
Mesure de l'équilibre des covariables entre groupes. SMD < 0.1 = bon équilibre.

### Matching
Appariement d'individus traités et contrôles similaires pour estimer l'effet causal.

### IPW (Inverse Probability Weighting)
Pondération par l'inverse du PS pour créer une pseudo-population équilibrée.

### Causal Forest
Algorithme de Machine Learning pour estimer des effets causaux hétérogènes (CATE).

---

## 📈 Applications pratiques

### Notebook LaLonde
- **Politique publique :** Évaluer l'efficacité des programmes de formation
- **Ciblage :** Identifier les profils bénéficiant le plus du programme
- **Optimisation :** Allouer les ressources aux individus avec CATE élevé

### Notebook A/B Test
- **Marketing :** Mesurer l'impact des campagnes publicitaires
- **Personnalisation :** Identifier les segments sensibles aux publicités
- **ROI :** Calculer le retour sur investissement par segment

---

## 🛠️ Installation

```bash
pip install pandas numpy scikit-learn matplotlib seaborn statsmodels networkx
pip install econml --no-binary econml
```

---

## 📝 Notes importantes

- Les deux notebooks incluent des visualisations détaillées (DAG, distributions, histogrammes)
- Les méthodes sont complémentaires : matching, IPW, et Causal Forest
- L'approche est rigoureuse avec diagnostics et validation
- Les CATE permettent de passer d'un effet moyen à une analyse personnalisée


---

## 📚 Références

- Dataset LaLonde : Matching::lalonde (statsmodels)
- EconML : Microsoft Research Library for Causal ML
- Méthodes d'inférence causale : Pearl, Rubin, Imbens
