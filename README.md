# Decade-Ahead Glacier Shrinkage Prediction Using Machine Learning

This repository contains a Jupyter Notebook implementation of an explainable machine learning framework for predicting glacier shrinkage in the Hindu Kush Himalaya (HKH). It uses glacier inventories from 1990, 2000, 2010, and 2020 to classify each matched glacier over the following ten years as **Stable**, **Shrinking**, or **Rapidly shrinking**.

The main presentation notebook is [`glacier_ml_assignment.ipynb`](glacier_ml_assignment.ipynb). 
## Project objective

The project aims to:

1. match the same glaciers across consecutive inventory years;
2. construct ten-year glacier-change observations without using future information as model input;
3. compare Logistic Regression, Decision Tree, Random Forest, and a Soft-Voting Ensemble; and
4. evaluate whether patterns learned from earlier decades can predict glacier change during 2010–2020.

## Dataset

The notebook uses ESRI Shapefile glacier inventories for four years:

| Inventory year | Original records |
| -------------: | ---------------: |
|           1990 |           65,188 |
|           2000 |           65,394 |
|           2010 |           64,885 |
|           2020 |           63,761 |

Each inventory includes glacier geometry and attributes such as area, elevation, slope, aspect, latitude, longitude, mountain range, and basin. A complete shapefile normally consists of related files such as `.shp`, `.dbf`, `.shx`, `.prj`, and `.cpg`; these files should remain together in the same folder.


> The glacier datasets may not be included in the GitHub repository because of their size or distribution conditions. Obtain the inventories separately from ICIMOD official site.

## How the notebook works

The notebook follows 13 sequential steps:

1. Imports the required Python libraries.
2. Loads the four glacier inventories with GeoPandas.
3. Audits record counts, missing values, duplicate identifiers, and invalid geometries.
4. Keeps the required attributes and removes unusable records.
5. Matches glaciers for 1990–2000, 2000–2010, and 2010–2020.
6. Calculates area change and assigns the three target classes.
7. Explores temporal and regional class distributions.
8. Selects only information available at the beginning of each prediction decade.
9. trains on the 1990–2000 and 2000–2010 transitions and reserves 2010–2020 for final testing.
10. Uses grouped cross-validation by mountain range and tunes the individual models.
11. selects soft-voting weights from out-of-fold training predictions.
12. Trains the final Soft-Voting Ensemble.
13. Evaluates decade-ahead predictions using metrics, classification reports, confusion matrices, and confidence values.

### Temporal glacier matching

Matching first uses unique `GLIMS_ID` values and verifies them spatially. Records that cannot be matched this way are considered for a conservative spatial fallback. A fallback pair must satisfy basin consistency, at least 50% overlap relative to the smaller polygon, a centroid distance of at most 5 km, and a one-to-one relationship. Possible splits and merges are diagnosed but are not forced into the modelling dataset.

This procedure produced the following matched ten-year observations:

| Transition      | Matched observations |
| --------------- | -------------------: |
| 1990–2000      |               63,300 |
| 2000–2010      |               62,909 |
| 2010–2020      |               61,476 |
| **Total** |    **187,685** |

### Target classes

The annualised glacier-area change is calculated as:

```text
Annual area change (%) = ((end area - start area) / start area × 100) / 10
```

The classes are then defined as:

| Class             | Annual area-change rule              |
| ----------------- | ------------------------------------ |
| Stable            | at least -0.5% per year              |
| Shrinking         | below -0.5% and above -1.0% per year |
| Rapidly shrinking | at most -1.0% per year               |

## Models and validation

The notebook compares:

- Logistic Regression
- Decision Tree
- Random Forest
- Soft-Voting Ensemble combining all three models

Missing numeric values are median-imputed and standardized. Missing categorical values are filled with the most frequent category and one-hot encoded. Model selection uses three-fold `GroupKFold` cross-validation based on mountain range. Macro F1 is the selection metric because the three change classes are imbalanced.

The Soft-Voting Ensemble combines predicted class probabilities. Its selected weights are **1:1:3**, giving one part each to Logistic Regression and Decision Tree and three parts to Random Forest. These weights are chosen only from grouped out-of-fold training predictions; the 2010–2020 test period is not used for model or weight selection.

## Final results

Performance on the untouched 2010–2020 period is:

| Model                |         Accuracy | Balanced accuracy |         Macro F1 |
| -------------------- | ---------------: | ----------------: | ---------------: |
| Logistic Regression  |           0.4825 |            0.5077 |           0.4528 |
| Decision Tree        |           0.5213 |            0.5034 |           0.4807 |
| Random Forest        | **0.5749** |            0.5252 |           0.5140 |
| Soft-Voting Ensemble |           0.5681 |  **0.5343** | **0.5164** |

Random Forest gives the highest overall accuracy, while the Soft-Voting Ensemble gives the best balanced accuracy and Macro F1. Therefore, the ensemble provides the strongest overall balance across all three classes. The moderate results also show that decade-ahead glacier classification is difficult when only inventory-derived geographic and topographic predictors are available.

## Requirements

- Python 3.10 or newer
- Jupyter Notebook or JupyterLab
- GeoPandas
- pandas
- NumPy
- Matplotlib
- Seaborn
- scikit-learn
- Joblib

Install the required packages with:

```bash
python -m pip install jupyter geopandas pandas numpy matplotlib seaborn scikit-learn joblib
```

GeoPandas also requires its geospatial dependencies. A Conda environment may be easier on systems where these dependencies are not already installed:

```bash
conda install -c conda-forge jupyter geopandas pandas numpy matplotlib seaborn scikit-learn joblib
```

## Running the notebook

1. Clone or download this repository.
2. Place all four glacier inventories in the folder structure shown above.
3. Open `glacier_ml_assignment.ipynb`.
4. Update `project_folder` and `source_folder` in the first code cell to match your computer.
5. Select **Kernel > Restart Kernel and Run All Cells**.

The matching and model-selection stages process large geospatial datasets and may take several minutes, depending on the available memory and CPU.

## Important interpretation

The model predicts a glacier's **change class for the next ten-year inventory period**. It does not predict an exact future glacier boundary or an exact future area. The predictions should be treated as screening-level evidence rather than operational forecasts because the dataset does not directly include climate variables such as temperature, precipitation, or debris-cover evolution.
