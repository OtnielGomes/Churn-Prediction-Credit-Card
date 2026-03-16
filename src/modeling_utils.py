# ================================================================================================= #
# >>> A module of functions and classes to facilitate the flow and creation of pipelines in sklearn #                                 
# ================================================================================================= #

# ======================================================== #
# Imports:                                                 #
# ======================================================== #
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import roc_auc_score
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFECV
from sklearn.utils.validation import check_is_fitted


# ======================================================== #
# Class FeatureEngineer                                    #
# ======================================================== #
import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer for the Telecom Churn dataset.

    This transformer creates:
    - total_spend
    - share_longmon, share_tollmon, share_equipmon, share_cardmon, share_wiremon
    - nonzero_count
    - flag_used_longmon, flag_used_tollmon, flag_used_equipmon, flag_used_cardmon, flag_used_wiremon
    - tenure_band
    - total_spend_per_tenure
    - tenure_x_internet, tenure_x_wireless, tenure_x_ebill, tenure_x_equip
    - total_services
    - legacy_bundle
    - legacy_bundle_no_internet
    - stability_index_raw
    - income_per_age
    - income_per_ed
    - ed_income_bucket
    - young_low_tenure
    - old_upper_tenure
    - toxic_score

    Leakage-sensitive thresholds are learned only on training data in fit():
    - age and tenure quartiles
    - income bin edges within each education group
    """

    def __init__(
        self,
        mon_cols=None,
        service_cols=None,
        tec_cols=None,
        eps=1e-6,
        tenure_bins=None,
        tenure_labels=None,
        ed_income_q=4,
        fill_value=0.0,
        return_copy=True
    ):
        self.mon_cols = mon_cols or ['longmon', 'tollmon', 'equipmon', 'cardmon', 'wiremon']
        self.service_cols = service_cols or [
            'ebill', 'equip', 'callcard', 'wireless', 'pager',
            'internet', 'voice', 'callwait', 'confer'
        ]
        self.tec_cols = tec_cols or ['internet', 'wireless', 'ebill', 'equip']
        self.toxic_cols = ['internet', 'wireless', 'equip', 'voice', 'pager']

        self.eps = eps
        self.tenure_bins = tenure_bins or [-np.inf, 3, 6, 12, 24, 48, np.inf]
        self.tenure_labels = tenure_labels or ['0-3', '4-6', '7-12', '13-24', '25-48', '49+']
        self.ed_income_q = ed_income_q
        self.fill_value = fill_value
        self.return_copy = return_copy

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_ = X.copy()

        if 'age' in X_.columns:
            self.age_q25_ = float(X_['age'].quantile(0.25))
            self.age_q75_ = float(X_['age'].quantile(0.75))
        else:
            self.age_q25_ = None
            self.age_q75_ = None

        if 'tenure' in X_.columns:
            self.tenure_q25_ = float(X_['tenure'].quantile(0.25))
            self.tenure_q75_ = float(X_['tenure'].quantile(0.75))
        else:
            self.tenure_q25_ = None
            self.tenure_q75_ = None

        self.ed_income_edges_ = {}

        if ('ed' in X_.columns) and ('income' in X_.columns):
            for ed_value, grp in X_.groupby('ed', dropna = False):
                s = grp['income'].astype('float64').dropna()

                if s.nunique() < 2:
                    self.ed_income_edges_[ed_value] = None
                    continue

                qs = np.linspace(0, 1, self.ed_income_q + 1)
                edges = s.quantile(qs).to_numpy()
                edges = np.unique(edges[~np.isnan(edges)])

                if len(edges) < 2:
                    self.ed_income_edges_[ed_value] = None
                else:
                    self.ed_income_edges_[ed_value] = edges.tolist()

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy() if self.return_copy else X

        try:
            # --------------------------
            # 1) Aggregations of costs
            # --------------------------
            existing_mon = [c for c in self.mon_cols if c in X.columns]

            if existing_mon:
                X['total_spend'] = X[existing_mon].sum(axis=1).astype('float32')
            else:
                X['total_spend'] = np.float32(0.0)

            denom = (X['total_spend'].astype('float32') + np.float32(self.eps))

            for col in self.mon_cols:
                if col in X.columns:
                    X[f'share_{col}'] = (X[col].astype('float32') / denom).astype('float32')
                else:
                    X[f'share_{col}'] = np.float32(0.0)

            if existing_mon:
                X['nonzero_count'] = (X[existing_mon] > 0).sum(axis=1).astype('int32')
            else:
                X['nonzero_count'] = np.int32(0)

            for col in self.mon_cols:
                if col in X.columns:
                    x = X[col].fillna(0)
                    X[f'flag_used_{col}'] = (x > 0).astype('int8')
                else:
                    X[f'flag_used_{col}'] = np.int8(0)

            # --------------------------
            # 2) Tenure features
            # --------------------------
            if 'tenure' in X.columns:
                X['tenure_band'] = pd.cut(
                    X['tenure'],
                    bins=self.tenure_bins,
                    labels=self.tenure_labels,
                    include_lowest=True
                )

                X['total_spend_per_tenure'] = (
                    X['total_spend'].astype('float32') /
                    (X['tenure'].fillna(0).astype('float32') + 1.0)
                ).astype('float32')
            else:
                X['tenure_band'] = pd.Series([pd.NA] * len(X), index=X.index)
                X['total_spend_per_tenure'] = np.float32(0.0)

            for col in self.tec_cols:
                if ('tenure' in X.columns) and (col in X.columns):
                    X[f'tenure_x_{col}'] = (
                        X['tenure'].fillna(0).astype('float32') *
                        X[col].fillna(0).astype('float32')
                    ).astype('int32')
                else:
                    X[f'tenure_x_{col}'] = np.int32(0)

            # --------------------------
            # 3) Portfolio / bundles
            # --------------------------
            for col in self.service_cols:
                if col not in X.columns:
                    X[col] = 0

            X[self.service_cols] = X[self.service_cols].fillna(0).astype('int32')
            X['total_services'] = X[self.service_cols].sum(axis=1).astype('int32')

            X['legacy_bundle'] = (
                (X['pager'] == 1) & (X['callcard'] == 1)
            ).astype('int8')

            X['legacy_bundle_no_internet'] = (
                (X['pager'] == 1) &
                (X['callcard'] == 1) &
                (X['internet'] == 0)
            ).astype('int8')

            # --------------------------
            # 4) Demographics / proxies
            # --------------------------
            if ('address' in X.columns) and ('employ' in X.columns):
                X['stability_index_raw'] = (
                    X['address'].fillna(0) + X['employ'].fillna(0)
                )
            else:
                X['stability_index_raw'] = np.int32(0)

            if ('income' in X.columns) and ('age' in X.columns):
                X['income_per_age'] = (
                    X['income'].fillna(0) / (X['age'].fillna(0) + 1)
                ).astype('float32')
            else:
                X['income_per_age'] = np.float32(0.0)

            if ('income' in X.columns) and ('ed' in X.columns):
                ed_safe = X['ed'].astype('float64').replace(0, np.nan)
                X['income_per_ed'] = (
                    X['income'].astype('float64') / ed_safe
                ).fillna(0).astype('float32')
            else:
                X['income_per_ed'] = np.float32(0.0)

            # --------------------------
            # 5) Income buckets within education
            # --------------------------
            if ('ed' in X.columns) and ('income' in X.columns) and hasattr(self, 'ed_income_edges_'):
                bucket = pd.Series(-1, index=X.index, dtype='int32')

                for ed_value, edges in self.ed_income_edges_.items():
                    if edges is None or len(edges) < 2:
                        continue

                    mask = X['ed'].eq(ed_value)
                    if not mask.any():
                        continue

                    cut_values = pd.cut(
                        X.loc[mask, 'income'].astype('float64'),
                        bins=edges,
                        labels=False,
                        include_lowest=True
                    )

                    bucket.loc[mask] = cut_values.fillna(-1).astype('int32')

                X['ed_income_bucket'] = bucket
            else:
                X['ed_income_bucket'] = pd.Series(-1, index=X.index, dtype='int32')

            # --------------------------
            # 6) Region flags learned on train only
            # --------------------------
            if (
                ('age' in X.columns) and
                ('tenure' in X.columns) and
                (self.age_q25_ is not None) and
                (self.tenure_q25_ is not None)
            ):
                X['young_low_tenure'] = (
                    (X['age'] < self.age_q25_) &
                    (X['tenure'] < self.tenure_q25_)
                ).astype('int8')
            else:
                X['young_low_tenure'] = np.int8(0)

            if (
                ('age' in X.columns) and
                ('tenure' in X.columns) and
                (self.age_q75_ is not None) and
                (self.tenure_q75_ is not None)
            ):
                X['old_upper_tenure'] = (
                    (X['age'] >= self.age_q75_) &
                    (X['tenure'] >= self.tenure_q75_)
                ).astype('int8')
            else:
                X['old_upper_tenure'] = np.int8(0)

            # --------------------------
            # 7) Risk feature
            # --------------------------
            existing_toxic = [c for c in self.toxic_cols if c in X.columns]
            if existing_toxic:
                X['toxic_score'] = X[existing_toxic].sum(axis=1).astype('int32')
            else:
                X['toxic_score'] = np.int32(0)

            # --------------------------
            # Final numeric cleanup
            # --------------------------
            num_cols = X.select_dtypes(include=[np.number]).columns
            X[num_cols] = (
                X[num_cols]
                .replace([np.inf, -np.inf], np.nan)
                .fillna(self.fill_value)
            )

            return X

        except Exception as e:
            raise RuntimeError(f'[FeatureEngineer] Error while creating features: {e}') from e



# ======================================================== #
# Class DtypeOptimizer                                     #
# ======================================================== #

class DtypeOptimizer(
    BaseEstimator,
    TransformerMixin
):
    """
    Transformer (scikit-learn) to standardize the column types (dtypes) of the Telecom dataset.

    Objective

    --------

    - Reduce memory usage (e.g., float32/int8).

    - Ensure consistency between training and testing (prevent columns from becoming objects).

    - Facilitate use in Pipeline/ColumnTransformer with models like LightGBM/XGBoost.

    Strategy

    ----------

    - continuous_cols -> float32

    - binary_cols -> int8 (0/1)

    - integer_cols -> int32 (counts/years/months; includes 'ed' if you want to treat it as a numeric ordinal)

    - categorical_cols -> category (nominals, e.g., 'custcat')

    Important Note

    ---------------------

    The `transform()` method expects a pandas.DataFrame with column names. If it receives another type,

    it attempts to convert it to DataFrame; this can create columns 0..N and you lose the original names,

    so it's best to use this step before any step that generates numpy arrays. [web:2]

    """


    # ======================================================== #
    # __Init__ - Function                                      #
    # ======================================================== #
    def __init__(
        self,
        categorical_cols:list = ['custcat'],
        integer_cols:list = ['tenure', 'age', 'address', 'employ', 'ed'],
        binary_cols:list = [
            'equip','callcard', 'wireless','ebill', 'voice', 'pager',
            'internet', 'callwait', 'confer',
        ],
        continuous_cols:list = [
            'income',  'longmon', 'tollmon', 'equipmon', 'cardmon','wiremon',
        ],
    ):
        
        """
        Parameters
        ----------
        categorical_cols : list of str
        Columns to be converted to dtype 'category' (nominal variables).

        integer_cols : list of str
        Columns to be converted to dtype 'int32' (counts/ages/time; numeric ordinals).

        binary_cols : list of str
        Binary columns (0/1) to be converted to dtype 'int8'.

        continuous_cols : list of str
        Continuous columns (e.g., monetary values) to be converted to dtype 'float32'.

        """

        self.categorical_cols = categorical_cols
        self.integer_cols = integer_cols
        self.binary_cols = binary_cols
        self.continuous_cols = continuous_cols


    # ======================================================== #
    # Fit - Function                                           #
    # ======================================================== #
    def fit(
        self,
        X, 
        y = None
    ):
        """
        Fit the transformer.

        Does not learn parameters; returns `self` to fulfill the scikit-learn contract

        and allow `fit_transform()`. [web:2]

        Parameters

        ---------
        X : pandas.DataFrame
        Input data.

        y : array-like, default=None
        Target (ignored).

        Returns

        -------
        self : DtypeOptimizer
        """
        return self
    

    # ======================================================== #
    # Transform - Function                                     #
    # ======================================================== #
    def transform(
        self, 
        X
    ):
        """
        Converts the dtypes of the columns defined in __init__.

        Parameters
        ----------
        X : pandas.DataFrame
        DataFrame with the dataset columns.

        Returns

        -------
        pandas.DataFrame
        Copy of the DataFrame with adjusted dtypes.
        """
        try:
            # Check Dataset
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFramme(X)
            X = X.copy()

            # Continuos
            continuous = [c for c in self.continuous_cols if c in X.columns]
            if continuous:
                X[continuous] = X[continuous].astype('float32')

            # Binary
            binary = [c for c in self.binary_cols if c in X.columns]
            if binary:
                X[binary] = X[binary].astype('int8')
            
            # Integer
            integer = [c for c in self.integer_cols if c in X.columns]
            if integer:
                X[integer] = X[integer].astype('int32')

            # Categorical
            categorical = [c for c in self.categorical_cols if c in X.columns]
            if categorical:
                X[categorical] = X[categorical].astype('category')
            
            return X

        except Exception as e:
            print(f'[Error] Failure to add data types to dataset variables: {str(e)}.')



# ======================================================== #
# _get_scores - Function                                   #
# ======================================================== #
def _get_scores(
    model,
    X, 
    pos_label = 1
):
    
    try:

        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)
            if proba.ndim == 2 and proba.shape[1] == 2:
                classes = getattr(model, 'classes_', None)
                if classes is None:
                    return proba[:, 1]
                pos_idx = int(np.where(classes == pos_label)[0][0])
                return proba[:, pos_idx]
            # If it's not binary (or doesn't have 2 columns), let the caller decide how to evaluate it
            raise ValueError('predict_proba no binary probabilities were returned (2 columns).')

        if hasattr(model, 'decision_function'):
            return model.decision_function(X)
        
        raise ValueError('The model does not expose predict_proba or decision_function; AUC may become invalid with predict().')
    
    except Exception as e:
            print(f'[Error] Failure to generate training scores: {str(e)}.')



# ======================================================== #
# classification_kfold_cv - Function                       #
# ======================================================== #
def classification_kfold_cv(
    models,
    X_train,
    y_train,
    n_folds: int = 5, 
    scoring: str = 'roc_auc', 
    pos_label: int = 1,
    n_jobs:int = None
):
    
    try:

        cv = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = 33)
        rows = []

        for name, estimator, in models.items():
            cv_out = cross_validate(
                estimator,
                X_train,
                y_train,
                scoring = scoring,
                cv = cv,
                return_train_score = False,
                n_jobs = n_jobs,
            )
            est_full = clone(estimator).fit(X_train, y_train)
            train_scores = _get_scores(est_full, X_train, pos_label = pos_label)
            train_roc_auc = roc_auc_score(y_train, train_scores)

            rows.append({
                'model': name,
                'avg_val_score': float(np.mean(cv_out['test_score'])),
                'val_score_std': float(np.std(cv_out['test_score'], ddof = 1)),
                'train_score': float(train_roc_auc),
                'avg_fit_time': float(np.mean(cv_out['fit_time'])),
            })

        return pd.DataFrame(rows).sort_values('avg_val_score', ascending = False).reset_index(drop = True)
    
    except Exception as e:
            print(f'[Error] Failure to execute cross-validation: {str(e)}.')




# ======================================================== #
# Class Recursive Feature Eliminator                       #
# ======================================================== #
class RecursiveFeatureEliminator(
    BaseEstimator, TransformerMixin
):
    """
    Select the most relevant features using RFECV.

    This transformer applies Recursive Feature Elimination with Cross-Validation
    (RFECV) to identify the subset of features that maximizes the selected
    scoring metric. It is compatible with the scikit-learn API and can be used
    inside pipelines.

    Parameters
    ----------
    estimator : estimator object, default=None
        Estimator used to compute feature importances.

        If None, a ``LGBMClassifier`` is used with ``verbosity=-1``.
    scoring : str, default='roc_auc'
        Scoring metric used to evaluate each subset of features during cross-validation.
    n_folds : int, default=5
        Number of folds used in ``StratifiedKFold``.
    step : int or float, default=1
        Number of features removed at each iteration if integer.
        If float between 0 and 1, it corresponds to the fraction of features
        removed at each iteration.
    random_state : int, default=33
        Random seed used in ``StratifiedKFold``.

    Attributes
    ----------
    rfe_ : RFECV
        Fitted RFECV object.
    feature_names_in_ : pandas.Index
        Names of the input features seen during fitting.
    selected_features_ : list of str
        Names of the selected features after RFECV.

    Notes
    -----
    When the input is a pandas DataFrame, the ``transform`` method preserves
    the original selected columns and their data types whenever possible.

    Examples
    --------
    > selector = RecursiveFeatureEliminator(scoring='roc_auc', n_folds=5)
    > selector.fit(X_train, y_train)
    > X_train_selected = selector.transform(X_train)
    > selector.selected_features_
    """
   

    # Initialize class
    def __init__(
        self,
        estimator = None,
        scoring = 'roc_auc',
        n_folds = 5,
        step = 1,
        random_state = 33,
    ):

        """
        Initialize the feature selector.

        Parameters
        ----------
        estimator : estimator object, default=None
            Estimator used to rank feature importance.
        scoring : str, default='roc_auc'
            Metric used by RFECV during feature selection.
        n_folds : int, default=5
            Number of folds used in cross-validation.
        step : int or float, default=1
            Step size for recursive feature elimination.
        random_state : int, default=33
            Random seed used in cross-validation splitting.
        """

        self.estimator = estimator
        self.scoring = scoring
        self.n_folds  = n_folds
        self.step = step        
        self.random_state = random_state

    
    # Function Fit
    def fit(
        self,
        X,
        y
    ):
        
        """
        Fit the RFECV selector to the input data.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Training features.
        y : array-like of shape (n_samples,)
            Target variable.

        Returns
        -------
        self : RecursiveFeatureEliminator
            Fitted transformer instance.

        Raises
        ------
        Exception
            Raised if the fitting process fails.
        """

        try:

            if self.estimator is None:
                estimator = LGBMClassifier(verbosity = -1)
            else:
                estimator = clone(self.estimator)
            
            cv = StratifiedKFold(
                n_splits = self.n_folds,
                shuffle = True,
                random_state = self.random_state
            )

            self.rfe_ = RFECV(
                estimator = estimator,
                step = self.step,
                cv = cv,
                scoring = self.scoring
            )

            self.rfe_.fit(X, y)

            if hasattr(X, 'columns'):
                self.feature_names_in_ = pd.Index(X.columns)
            else:
                self.feature_names_in_ = pd.Index(
                    [f'x{i}' for i in range(self.rfe_.n_features_in_)]
                )
            
            self.selected_features_ = list(
                self.feature_names_in_[self.rfe_.support_]
            )

            return self
        
        except Exception as e:
            print(f'[Error] Failure to execute fit function RecursiveFeatureEliminator: {str(e)}.')

    # Transform
    def transform(
        self,
        X
    ):
        
        """
        Transform ``X`` by keeping only the selected features.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Input data to transform.

        Returns
        -------
        pandas.DataFrame
            DataFrame containing only the selected features.

        Raises
        ------
        Exception
            Raised if the transform process fails.
        """
        
        try:

            check_is_fitted(self, 'rfe_')
            X_selected = self.rfe_.transform(X)

            if hasattr(X, 'loc'):
                return X.loc[:, self.selected_features_].copy()

            X_selected = self.rfe_.transform(X)
            return pd.DataFrame(X_selected, columns = self.selected_features_)
        
        except Exception as e:
            print(f'[Error] Failure to execute fit function RecursiveFeatureEliminator: {str(e)}.')