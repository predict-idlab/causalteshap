	
<p align="center">
    <a href="#readme">
        <img alt="Causalteshap logo" src="https://github.com/predict-idlab/causalteshap/blob/main/causalteshap_logo.PNG" width=70%>
    </a>
</p>


> *causalteshap* is a **treatment effect analysis** method that uses statistical hypothesis testing on **Shapley values** to find predictive and prognostic features.  

## Installation ⚙️

UNDER CONSTRUCTION
| [**pip**](https://pypi.org/project/causalteshap/) | `pip install causalteshap` | 
| ---| ----|

## Usage 🛠

*causalteshap* is built to be intuitive, it supports various tree-based models for its S-learner for classification and regression tasks.  
<!-- It is also implemented as sklearn `Transformer` component, allowing convenient integration in `sklearn` pipelines. -->

```py
from causalteshap.causalteshap import CausalteShap
from catboost import CatBoostClassifier

X, T, y = ...  # your classification dataset with treatment T

causalteshap_object = CausalteShap(
    model=CatBoostClassifier(n_estimators=250, verbose=0, use_best_model=True,cat_features=["T"], meta_learner="S")
)

causalteshap_object.fit(X, T, y)  # Fit the PowerShap feature selector
causalteshap_object.get_analysis_df()  # Reduce the dataset to the selected features

```

## Features ✨

* default mode
* `scikit-learn` compatible
* supports various models
* Works for S-learners
* insights into the meta-learners features

## Benchmarks ⏱

UNDER CONSTRUCTION
Check out our benchmark results [here](examples/results/).  

## How does it work ⁉️

UNDER CONSTRUCTION

## Referencing our package :memo:

If you use *causalteshap* in a scientific publication, we would highly appreciate citing us as:

UNDER CONSTRUCTION

---

<p align="center">
👤 <i>Jarne Verhaeghe</i>
</p>

## License
This package is available under the MIT license. More information can be found here: https://github.com/predict-idlab/causalteshap/blob/main/LICENSE
