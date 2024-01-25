	
<p align="center">
    <a href="#readme">
        <img alt="Causalteshap logo" src="https://github.com/predict-idlab/causalteshap/blob/main/causalteshap_logo.PNG" width=70%>
    </a>
</p>


> *causalteshap* is a **treatment effect analysis** method that uses statistical hypothesis testing on **Shapley values** to find predictive and prognostic features.  

## Installation ⚙️

UNDER CONSTRUCTION, PIP DOES NOT WORK YET
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

Check out our benchmark results [here](examples/results/figures/).  

## How does it work ⁉️

Causalteshap uses an introduced noise feature and statistical tests to determine whether a feature is prognostic (i.e. only contributing to the output) or predictive (i.e. explaining the effect of a treatment). First, we train an S-learner on the data. Then we use Shapley values to explain the attribution of the features of the S-learner, into two cases, one where the treatment is set to 0 ($S_0$) and one where it is set to 1 ($S_1$), including the introduced noise feature. Then the S-learner Shapley values are $S = S_1 - S_0$. Ideally, a purely prognostic feature X will have a S equal to zero. However, the SHAP library, especially treeSHAP, tends to also attribute importance to noise. Therefore, simply comparing $S$ to zero is not guaranteed to work. Therefore,  CausalteShap uses a two-part approach to deal with noise:
- If the feature is purely prognostic, then the $S_0$ and $S_1$ distribution should have the same variance and same mean. This is done using both the Fligner and the student t-test with unequal variance.
- When these distributions are different and the feature is truly prognostic, then $|S(X_{noisy}|$ of a known noise variable $X_{noisy}$ that contains no information should be larger or equal compared to $|S_{0}(X)|$. This covers the cases where these differences would be caused by noise. This is done using the Kolmogorov-Smirnov test.

If a feature passes both parts, i.e. significant result on both the KS-test and either the t-test or Fligner test (that tests whether either the mean and or variance is different), we determine the feature to be predictive. In the case any of the parts fail, the feature is flagged as prognostic.

## Referencing our package :memo:

If you use *causalteshap* in a scientific publication, we would highly appreciate citing us.

SCIENTIFIC PAPER UNDER REVIEW

---

<p align="center">
👤 <i>Jarne Verhaeghe</i>
</p>

## License
This package is available under the MIT license. More information can be found here: https://github.com/predict-idlab/causalteshap/blob/main/LICENSE
