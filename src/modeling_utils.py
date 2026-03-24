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
#from __future__ import annotations
import math
from typing import Any, Hashable, Sequence
import numpy.typing as npt
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    brier_score_loss,
    classification_report
)



# ======================================================== #
# Class FeatureEngineer                                    #
# ======================================================== #
class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom feature engineering transformer for a credit card churn dataset.

    This transformer creates ratio, interaction, binary, and ordinal features
    from customer transaction, balance, tenure, and activity variables. It is
    designed to be compatible with scikit-learn pipelines.

    Created features
    ----------------
    Ratio / percentage features:
    - avg_ticket
    - inactive_per_tenure

    Interaction / combination features:
    - total_spending
    - change_gap_amt_ct
    - utilization_amont
    - credit_revolving

    Binary features / flags:
    - months_inactive_12_mon_0
    - contacts_count_12_mon_0
    - dependent_count_0
    - total_revolving_bal_0
    - total_amt_chng_q4_q1_0
    - total_ct_chng_q4_q1_0
    - avg_utilization_ratio_0
    - low_activity_low_value_flag
    - is_pos_graduate
    - total_amt_q75
    - total_ct_q75

    Bucket / ordinal features:
    - trans_ct_bucket
    - trans_amt_bucket
    - tenure_bucket

    Notes
    -----
    The quantile boundaries used to build `low_activity_low_value_flag` are
    learned only during `fit()` from the training data and reused in
    `transform()`. This helps prevent leakage between training and validation
    or test sets.

    Infinite values are replaced and numeric missing values are filled during
    the final cleanup step.

    Parameters
    ----------
    quantiles : int, default=4
        Number of quantile groups used to estimate the thresholds for
        `low_activity_low_value_flag`.

    fill_value : float, default=0.0
        Value used to fill missing numeric values after feature generation.

    return_copy : bool, default=True
        If True, returns a copy of the input DataFrame. If False, transformations
        are applied in place when possible.

    Attributes
    ----------
    util_bins_ : ndarray or None
        Quantile bin edges learned from `avg_utilization_ratio`.

    revol_bins_ : ndarray or None
        Quantile bin edges learned from `total_revolving_bal`.

    amt_bins_ : ndarray or None
        Quantile bin edges learned from `total_trans_amt`.

    relation_bins_ : ndarray or None
        Quantile bin edges learned from `total_relationship_count`.

    trans_bins_ : ndarray or None
        Quantile bin edges learned from `total_trans_ct`.
    """

    def __init__(
        self,
        quantiles = 4,
        fill_value = 0.0,
        return_copy = True
    ):
        self.quantiles = quantiles
        self.fill_value = fill_value
        self.return_copy = return_copy

    def fit(self, X, y = None):

        """
        Learn quantile bin edges from the training data.

        This method estimates the quantile boundaries required to create the
        `low_activity_low_value_flag` feature. The learned boundaries are stored
        as fitted attributes and later reused in `transform()` so that the same
        thresholds are applied consistently to new data.

        Parameters
        ----------
        X : pandas.DataFrame or array-like of shape (n_samples, n_features)
            Input data used to learn quantile thresholds.

        y : array-like of shape (n_samples,), default=None
            Ignored. Present only for scikit-learn API compatibility.

        Returns
        -------
        self : FeatureEngineer
            Fitted transformer.
        """
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X_ = X.copy()

        # Save quantiles only from training
        self.util_bins_ = None
        self.revol_bins_ = None
        self.amt_bins_ = None
        self.relation_bins_ = None
        self.trans_bins_ = None

        # It only calculates if the columns exist.
        if 'avg_utilization_ratio' in X_.columns:
            self.util_bins_ = pd.qcut(
                X_['avg_utilization_ratio'],
                q = self.quantiles,
                duplicates = 'drop',
                retbins = True
            )[1]

        if 'total_revolving_bal' in X_.columns:
            self.revol_bins_ = pd.qcut(
                X_['total_revolving_bal'],
                q = self.quantiles,
                duplicates = 'drop',
                retbins = True
            )[1]

        if 'total_trans_amt' in X_.columns:
            self.amt_bins_ = pd.qcut(
                X_['total_trans_amt'],
                q = self.quantiles,
                duplicates = 'drop',
                retbins = True
            )[1]

        if 'total_relationship_count' in X_.columns:
            self.relation_bins_ = pd.qcut(
                X_['total_relationship_count'],
                q = self.quantiles,
                duplicates = 'drop',
                retbins = True
            )[1]

        if 'total_trans_ct' in X_.columns:
            self.trans_bins_ = pd.qcut(
                X_['total_trans_ct'],
                q = self.quantiles,
                duplicates = 'drop',
                retbins = True
            )[1]

        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)

        X = X.copy() if self.return_copy else X

        try:
            # =========================================================
            # A. Percentage/Ratio Features
            # =========================================================
            if {'total_trans_amt', 'total_trans_ct'}.issubset(X.columns):
                X['avg_ticket'] = (
                    X['total_trans_amt'].astype('float64') /
                    X['total_trans_ct'].replace(0, np.nan).astype('float64')
                ).astype('float32')
            else:
                X['avg_ticket'] = np.float32(0.0)


            if {'months_on_book', 'months_inactive_12_mon'}.issubset(X.columns):
                X['inactive_per_tenure'] = (
                    X['months_on_book'].astype('float64') /
                    X['months_inactive_12_mon'].replace(0, np.nan).astype('float64')
                ).astype('float32')
            else:
                X['inactive_per_tenure'] = np.float32(0.0)

            # =========================================================
            # B. Interaction/Combination Features
            # =========================================================
            if {'total_trans_amt', 'total_revolving_bal'}.issubset(X.columns):
                X['total_spending'] = (
                    X['total_trans_amt'].astype('float64') +
                    X['total_revolving_bal'].astype('float64')
                ).astype('float32')
            else:
                X['total_spending'] = np.float32(0.0)


            if {'total_amt_chng_q4_q1', 'total_ct_chng_q4_q1'}.issubset(X.columns):
                X['change_gap_amt_ct'] = (
                    X['total_amt_chng_q4_q1'].astype('float64') +
                    X['total_ct_chng_q4_q1'].astype('float64')
                ).astype('float32')
            else:
                X['change_gap_amt_ct'] = np.float32(0.0)


            if {'credit_limit', 'avg_utilization_ratio'}.issubset(X.columns):
                X['utilization_amont'] = (
                    X['credit_limit'].astype('float64') *
                    X['avg_utilization_ratio'].astype('float64')
                ).astype('float32')
            else:
                X['utilization_amont'] = np.float32(0.0)


            if {'credit_limit', 'total_revolving_bal'}.issubset(X.columns):
                X['credit_revolving'] = (
                    X['credit_limit'].astype('float64') -
                    X['total_revolving_bal'].astype('float64')
                ).astype('float32')
            else:
                X['credit_revolving'] = np.float32(0.0)

            # =========================================================
            # C. Binary Features / Flags
            # =========================================================
            for col in [
                'months_inactive_12_mon',
                'contacts_count_12_mon',
                'dependent_count',
                'total_revolving_bal',
                'total_amt_chng_q4_q1',
                'total_ct_chng_q4_q1',
                'avg_utilization_ratio'
            ]:
                if col in X.columns:
                    X[f'{col}_0'] = (X[col] == 0).astype('int8')
                else:
                    X[f'{col}_0'] = np.int8(0)

            # low_activity_low_value_flag with bins learned in fit
            def _quantile_index(series, bins):
                if (series is None) or (bins is None):
                    return None
                return pd.cut(
                    series,
                    bins = bins,
                    labels = False,
                    include_lowest = True
                )

            util_q = _quantile_index(
                X['avg_utilization_ratio'] if 'avg_utilization_ratio' in X.columns else None,
                self.util_bins_
            )
            revol_q = _quantile_index(
                X['total_revolving_bal'] if 'total_revolving_bal' in X.columns else None,
                self.revol_bins_
            )
            amt_q = _quantile_index(
                X['total_trans_amt'] if 'total_trans_amt' in X.columns else None,
                self.amt_bins_
            )
            relation_q = _quantile_index(
                X['total_relationship_count'] if 'total_relationship_count' in X.columns else None,
                self.relation_bins_
            )
            trans_q = _quantile_index(
                X['total_trans_ct'] if 'total_trans_ct' in X.columns else None,
                self.trans_bins_
            )

            if all(q is not None for q in [util_q, revol_q, amt_q, relation_q, trans_q]):
                X['low_activity_low_value_flag'] = (
                    (util_q == util_q.min()) &
                    (trans_q == trans_q.min()) &
                    (amt_q == amt_q.min()) &
                    (revol_q == revol_q.min()) &
                    (relation_q == relation_q.min())
                ).astype('int8')
            else:
                X['low_activity_low_value_flag'] = np.int8(0)

            # is_pos_graduate
            if 'education_level' in X.columns:
                X['is_pos_graduate'] = (
                    (X['education_level'] == 'Doctorate') |
                    (X['education_level'] == 'Post-Graduate')
                ).astype('int8')
            else:
                X['is_pos_graduate'] = np.int8(0)

            # total_amt_q75 e total_ct_q75 
            if 'total_amt_chng_q4_q1' in X.columns:
                X['total_amt_q75'] = (X['total_amt_chng_q4_q1'] >= 0.75).astype('int8')
            else:
                X['total_amt_q75'] = np.int8(0)

            if 'total_ct_chng_q4_q1' in X.columns:
                X['total_ct_q75'] = (X['total_ct_chng_q4_q1'] >= 0.75).astype('int8')
            else:
                X['total_ct_q75'] = np.int8(0)

            # =========================================================
            # D. Buckets / Ordinal Segmentation
            # =========================================================
            if 'total_trans_ct' in X.columns:
                X['trans_ct_bucket'] = pd.cut(
                    X['total_trans_ct'],
                    bins = [0, 25, 50, 75, 100, np.inf],
                    labels = [
                        'very_low_activity',
                        'low_activity',
                        'medium_activity',
                        'high_activity',
                        'very_high_activity'
                    ],
                    include_lowest=True,
                    ordered=True
                )
            else:
                X['trans_ct_bucket'] = pd.Series(
                    pd.Categorical(
                        [np.nan] * len(X),
                        categories = [
                            'very_low_activity',
                            'low_activity',
                            'medium_activity',
                            'high_activity',
                            'very_high_activity'
                        ],
                        ordered = True
                    ),
                    index = X.index
                )

            if 'total_trans_amt' in X.columns:
                X['trans_amt_bucket'] = pd.cut(
                    X['total_trans_amt'],
                    bins = [0, 2500, 5000, 7500, 10000, np.inf],
                    labels = [
                        'very_low_activity',
                        'low_activity',
                        'medium_activity',
                        'high_activity',
                        'very_high_activity'
                    ],
                    include_lowest = True,
                    ordered = True
                )
            else:
                X['trans_amt_bucket'] = pd.Series(
                    pd.Categorical(
                        [np.nan] * len(X),
                        categories = [
                            'very_low_activity',
                            'low_activity',
                            'medium_activity',
                            'high_activity',
                            'very_high_activity'
                        ],
                        ordered = True
                    ),
                    index = X.index
                )

            if 'months_on_book' in X.columns:
                X['tenure_bucket'] = pd.cut(
                    X['months_on_book'],
                    bins=[0, 12, 24, 48, np.inf],
                    labels=['new', 'developing', 'established', 'loyal'],
                    include_lowest = True,
                    ordered = True
                )
            else:
                X['tenure_bucket'] = pd.Series(
                    pd.Categorical(
                        [np.nan] * len(X),
                        categories = ['new', 'developing', 'established', 'loyal'],
                        ordered = True
                    ),
                    index=X.index
                )

            # =========================================================
            # Final cleaning of numeric values
            # =========================================================
            num_cols = X.select_dtypes(include = [np.number]).columns
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
        categorical_cols:list = [
            'Gender', 'Education_Level', 'Marital_Status', 'Income_Category', 'Card_Category', 'Attrition_Flag'
        ],
        integer_cols:list = [
            'Customer_Age', 'Dependent_count', 'Months_on_book', 'Total_Relationship_Count', 'Months_Inactive_12_mon','Contacts_Count_12_mon', 'Total_Trans_Ct',
        ],
        continuous_cols:list = [
            'Credit_Limit', 'Total_Revolving_Bal','Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio'
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

        and allow `fit_transform()`

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

            # Integer
            integer = [c for c in self.integer_cols if c in X.columns]
            if integer:
                X[integer] = X[integer].astype('int32')

            # Categorical
            categorical = [c for c in self.categorical_cols if c in X.columns]
            if categorical:
                X[categorical] = X[categorical].astype('category')
            
            # Adjusting column names to standard lowercase letters.
            X = X.rename(columns = str.lower)
            
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
            one_hot_cols = [
                'gender', 'marital_status', 
            ]
            est_full = clone(estimator).fit(X_train, y_train,)
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



# ======================================================== #
# Function models_performance_barplots                     #
# ======================================================== #
def models_performance_barplots(
    data: pd.DataFrame,
    models_col: str,
    palette: Sequence[Any] | None = None,
    title: str = 'Models Performance Comparison',
    num_columns: int = 1,
    figsize_per_row: float = 5
):
    """
    Generate bar plots to compare the performance of multiple models across metrics.

    Parameters
    ----------
    data : pd.DataFrame
        DataFrame containing one column with model identifiers and one or more
        numeric metric columns.

    models_col : str
        Name of the column containing model identifiers.

    palette : Sequence[Any] | None, default=None
        Color palette used in the plots. If None, a viridis palette is created
        automatically based on the number of unique models.

    title : str, default='Models Performance Comparison'
        Main title of the figure.

    num_columns : int, default=1
        Number of subplot columns.

    figsize_per_row : float, default=5
        Height of each subplot row in inches.

    Returns
    -------
    NONE

    Raises
    ------
    TypeError
        If `data` is not a pandas DataFrame.

    ValueError
        If `models_col` is not present in `data`.

    ValueError
        If `num_columns` is less than 1.

    ValueError
        If no metric columns are found.

    ValueError
        If any metric column is not numeric.
    """
    if not isinstance(data, pd.DataFrame):
        raise TypeError('data must be a pandas DataFrame.')

    if models_col not in data.columns:
        raise ValueError(f'Column \'{models_col}\' was not found in data.')

    if num_columns < 1:
        raise ValueError('num_columns must be greater than or equal to 1.')

    metric_cols = [col for col in data.columns if col != models_col]

    if not metric_cols:
        raise ValueError('No metric columns were found to plot.')

    non_numeric_cols = [
        col for col in metric_cols
        if not pd.api.types.is_numeric_dtype(data[col])
    ]
    if non_numeric_cols:
        raise ValueError(
            f'All metric columns must be numeric. Non-numeric columns found: {non_numeric_cols}'
        )

    n_models = data[models_col].nunique()
    n_metrics = len(metric_cols)
    num_rows = math.ceil(n_metrics / num_columns)

    if palette is None:
        palette = sns.color_palette('viridis', n_models)

    fig_width = 15 * num_columns
    fig_height = figsize_per_row * num_rows
    fig, axes = plt.subplots(
        num_rows,
        num_columns,
        figsize = (fig_width, fig_height),
        squeeze = False
    )
    axes = axes.flatten()

    for i, column in enumerate(metric_cols):
        ax = axes[i]

        barplot = sns.barplot(
            data = data,
            x = models_col,
            y = column,
            hue = models_col,
            dodge = False,
            edgecolor = 'black',
            saturation = 1,
            palette = palette,
            ax = ax,
            legend = False
        )

        ax.set_title(column, fontsize = 18, fontweight = 'bold')
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.tick_params(axis = 'x', labelsize = 12, rotation = 0)
        ax.tick_params(axis = 'y', labelsize = 11)
        ax.grid(axis = 'y', linestyle = '--', alpha = 0.3)
        ax.grid(axis = 'x', linestyle = '--', alpha = 0.3)
        ax.set_yticklabels([])

        for patch in barplot.patches:
            height = patch.get_height()

            if pd.isna(height):
                continue

            ax.annotate(
                f'{height:.4f}',
                (
                    patch.get_x() + patch.get_width() / 2.0,
                    height / 1.06 if height != 0 else height + 0.001
                ),
                ha = 'center',
                va = 'top' if height > 0 else 'bottom',
                fontsize = 12,
                fontweight = 'bold',
                color = 'white'
            )

    for j in range(n_metrics, len(axes)):
        fig.delaxes(axes[j])

    fig.suptitle(title, fontsize = 22, fontweight = 'bold', y = 1.02)
    plt.tight_layout()
    plt.show()


# ======================================================== #
# Function plot_kde_predictions                            #
# ======================================================== #
def plot_kde_predictions(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    palette: Sequence[str] = ('#12e193', '#feb308'),
    title: str = 'Prediction Probabilities',
    n_bins: int = 10,
    random_state: int = 33,
    figsize: tuple[float, float] = (12, 8)
) -> pd.DataFrame:
    """
    Plot:
    1. KDE distribution of predicted probabilities by class.
    2. Score ordering by quantile bin with observed event rate.

    Parameters
    ----------
    y_true : npt.ArrayLike
        Binary ground-truth labels.

    y_score : npt.ArrayLike
        Predicted probabilities or scores for the positive class.

    palette : Sequence[str], default=('#12e193', '#feb308')
        Colors used in the KDE plot by class.

    title : str, default='Prediction Probabilities'
        Title of the KDE plot.

    n_bins : int, default=10
        Number of quantile bins used to segment the scores.

    random_state : int, default=33
        Seed used to generate small random noise for tie breaking before `qcut`.

    figsize : tuple[float, float], default=(12, 8)
        Figure size in inches, given as (width, height).

    Returns
    -------
    pd.DataFrame
        DataFrame containing the bin index and observed event rate.

    Raises
    ------
    ValueError
        If `y_true` and `y_score` have different lengths.

    ValueError
        If the target is not binary.

    ValueError
        If no valid rows remain after cleaning.

    RuntimeError
        If any error occurs during plotting.
    """
    try:
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()

        if len(y_true) != len(y_score):
            raise ValueError('y_true and y_score must have the same length.')

        if len(np.unique(y_true)) != 2:
            raise ValueError('This function only supports binary classification.')

        plot_df = pd.DataFrame({
            'y_true': y_true,
            'y_score': y_score
        }).dropna().copy()

        if plot_df.empty:
            raise ValueError('There are no valid rows after removing missing values.')

        plot_df['y_true'] = pd.to_numeric(plot_df['y_true'], errors = 'coerce')
        plot_df['y_score'] = pd.to_numeric(plot_df['y_score'], errors = 'coerce')
        plot_df = plot_df.dropna().copy()

        if plot_df.empty:
            raise ValueError('The dataset became empty after numeric conversion.')

        rng = np.random.default_rng(random_state)
        noise = rng.uniform(0, 1e-4, size = len(plot_df))
        plot_df['score_adj'] = plot_df['y_score'].to_numpy() + noise

        plot_df = plot_df.sort_values(by = 'score_adj', ascending = True).reset_index(drop = True)

        plot_df['decile'] = pd.qcut(
            plot_df['score_adj'],
            q = n_bins,
            labels = False,
            duplicates = 'drop'
        )

        decile_df = (
            plot_df
            .groupby('decile', observed = True)['y_true']
            .mean()
            .reset_index()
            .rename(columns = {'y_true': 'event_rate'})
        )

        decile_df['decile'] = decile_df['decile'].astype(int) + 1

        plt.rc('font', size = 10)
        fig, axes = plt.subplots(
            2, 1,
            figsize = figsize,
            gridspec_kw = {'height_ratios': [1.2, 1]}
        )

        ax_kde, ax_bar = axes

        sns.kdeplot(
            data = plot_df,
            x = 'y_score',
            hue = 'y_true',
            fill = True,
            alpha = 0.4,
            bw_adjust = 1,
            palette = palette,
            linewidth = 1,
            cut = 0,
            ax = ax_kde
        )

        ax_kde.set_title(title, fontsize = 14, fontweight = 'bold')
        ax_kde.set_xlabel('Predicted probability')
        ax_kde.set_ylabel(None)
        ax_kde.set_yticklabels([])
        ax_kde.grid(axis = 'y', linestyle = '--', linewidth = 0.3)
        ax_kde.grid(axis = 'x', linestyle = '--', linewidth = 0.3)
        sns.despine(ax = ax_kde, top = True, right = True, left = True, bottom = False)

        bars = ax_bar.bar(
            decile_df['decile'],
            decile_df['event_rate'],
            color = 'darkorange',
            edgecolor = 'white',
            linewidth = 0.8
        )

        ax_bar.set_title(
            'Probability scores ordering - Event rate per decile',
            loc = 'left',
            fontweight = 'bold',
            fontsize = 13
        )
        ax_bar.set_xlabel('Decile', labelpad = 12)
        ax_bar.set_ylabel(None)
        ax_bar.set_xticks(decile_df['decile'])
        ax_bar.tick_params(axis = 'both', which = 'both', length = 0)
        ax_bar.yaxis.set_visible(False)

        ax_bar.spines['top'].set_visible(False)
        ax_bar.spines['right'].set_visible(False)
        ax_bar.spines['left'].set_visible(False)
        ax_bar.grid(False)

        offset = max(decile_df['event_rate'].max() * 0.03, 0.005)

        for bar, event_rate in zip(bars, decile_df['event_rate']):
            height = bar.get_height()
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2,
                height + offset,
                f'{event_rate * 100:.1f}%',
                ha = 'center',
                va = 'bottom',
                color = 'black',
                fontsize = 10,
                fontweight = 'bold'
            )

        plt.tight_layout()
        plt.show()

        return decile_df

    except Exception as e:
        raise RuntimeError(
            f'Failed to generate prediction probability graphs: {str(e)}.'
        ) from e



# ======================================================== #
# Function evaluation_scores                               #
# ======================================================== #
def evaluation_scores(
    y_true: npt.ArrayLike,
    y_score: npt.ArrayLike,
    X: pd.DataFrame | npt.ArrayLike | None = None,
    pos_label: Hashable = 1,
    threshold: float = 0.5,
    figsize: tuple[float, float] = (6, 12),
    title_prefix: str = ''
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate a binary classification model using true labels and predicted scores.

    The function computes both threshold-based metrics and ranking-based metrics,
    and generates three visualizations:
    1. ROC curve
    2. Precision-Recall curve
    3. Confusion matrix

    Parameters
    ----------
    y_true : npt.ArrayLike
        Ground-truth target values.

    y_score : npt.ArrayLike
        Predicted scores for the positive class.
        Ideally, predicted probabilities or monotonic model scores.

    X : pd.DataFrame | npt.ArrayLike | None, default=None
        Optional parameter kept for future compatibility.
        It is currently not used in the function.

    pos_label : Hashable, default=1
        Label considered as the positive class.

    threshold : float, default=0.5
        Cutoff used to convert `y_score` into binary predictions.

    figsize : tuple[float, float], default=(6, 12)
        Figure size in inches, given as (width, height).

    title_prefix : str, default=''
        Prefix added to subplot titles.

    Returns
    -------
    tuple[pd.DataFrame, pd.DataFrame]
        A tuple containing:
        - report_df: Classification report by class and aggregate rows.
        - metrics_df: Summary metrics DataFrame.

    Raises
    ------
    ValueError
        If `y_true` and `y_score` have different lengths.

    ValueError
        If the problem is not binary.

    RuntimeError
        If any failure occurs while computing metrics or generating the plots.

    Notes
    -----
    - This function assumes binary classification.
    - Brier score is most appropriate when `y_score` represents probabilities.
    - The `X` parameter is currently unused.
    """
    try:
        y_true = np.asarray(y_true).ravel()
        y_score = np.asarray(y_score).ravel()

        if len(y_true) != len(y_score):
            raise ValueError('y_true and y_score need to be the same size.')

        unique_labels = np.unique(y_true)
        if len(unique_labels) != 2:
            raise ValueError(
                f'The function only supports binary classification. Labels found: {unique_labels}'
            )

        y_bin = (y_true == pos_label).astype(int)

        fpr, tpr, _ = roc_curve(y_bin, y_score)
        auroc_value = roc_auc_score(y_bin, y_score)

        precision_vals, recall_vals, _ = precision_recall_curve(y_bin, y_score)
        ap_value = average_precision_score(y_bin, y_score)

        prevalence = y_bin.mean()

        y_pred = (y_score >= threshold).astype(int)

        cm = confusion_matrix(y_bin, y_pred)

        accuracy = accuracy_score(y_bin, y_pred)
        precision = precision_score(y_bin, y_pred, zero_division = 0)
        recall = recall_score(y_bin, y_pred, zero_division = 0)
        f1 = f1_score(y_bin, y_pred, zero_division = 0)

        roc_auc = auroc_value
        pr_auc = ap_value
        brier = brier_score_loss(y_bin, y_score)
        ks = np.max(tpr - fpr)
        gini = 2 * roc_auc - 1

        report_dict = classification_report(
            y_bin,
            y_pred,
            output_dict = True,
            zero_division = 0
        )
        report_df = pd.DataFrame(report_dict).T.reset_index()
        report_df = report_df.rename(columns = {'index': 'class'})
        report_df = report_df.round(4)

        desired_order = ['0', '1', 'accuracy', 'macro avg', 'weighted avg']
        report_df['order'] = report_df['class'].astype(str).apply(
            lambda x: desired_order.index(x) if x in desired_order else 999
        )

        report_df = (
            report_df
            .sort_values('order')
            .drop(columns = 'order')
            .reset_index(drop = True)
        )

        metrics_df = pd.DataFrame({
            'Metric': [
                'Accuracy',
                'Precision',
                'Recall',
                'F1-Score',
                'ROC-AUC',
                'KS',
                'Gini',
                'PR-AUC',
                'Brier'
            ],
            'Value': [
                accuracy,
                precision,
                recall,
                f1,
                roc_auc,
                ks,
                gini,
                pr_auc,
                brier
            ]
        })

        metrics_df['Value'] = metrics_df['Value'].round(4)

        fig = plt.figure(figsize = figsize)
        gs = GridSpec(3, 1, figure = fig, height_ratios = [1, 1, 1.15])

        ax_roc = fig.add_subplot(gs[0, 0])
        ax_pr = fig.add_subplot(gs[1, 0])
        ax_cm = fig.add_subplot(gs[2, 0])

        ax_roc.plot(
            fpr, tpr,
            color = '#1f77b4',
            lw = 2,
            label = f'ROC (AUROC = {auroc_value:.3f})'
        )
        ax_roc.plot(
            [0, 1], [0, 1],
            color = 'gray',
            linestyle = '--',
            label = 'Random'
        )
        ax_roc.set_xlabel('False Positive Rate', fontsize = 12)
        ax_roc.set_ylabel('True Positive Rate', fontsize = 12)
        ax_roc.set_title(f'{title_prefix}ROC Curve', fontsize = 14, fontweight = 'bold')
        ax_roc.legend(loc = 'lower right', fontsize = 10)
        ax_roc.grid(True, linestyle = '--', linewidth = 0.5)

        ax_pr.plot(
            recall_vals, precision_vals,
            color = 'darkorange',
            lw = 2,
            label = f'PR Curve (AP = {ap_value:.3f})'
        )
        ax_pr.axhline(
            prevalence,
            color = 'gray',
            linestyle = '--',
            label = f'Baseline = {prevalence:.3f}'
        )
        ax_pr.set_xlabel('Recall', fontsize = 12)
        ax_pr.set_ylabel('Precision', fontsize = 12)
        ax_pr.set_title(
            f'{title_prefix}Precision-Recall Curve',
            fontsize = 14,
            fontweight = 'bold'
        )
        ax_pr.legend(loc = 'upper right', fontsize = 10)
        ax_pr.grid(True, linestyle = '--', linewidth = 0.5)

        disp = ConfusionMatrixDisplay(
            confusion_matrix = cm,
            display_labels = [0, 1]
        )
        disp.plot(ax = ax_cm, cmap = 'viridis', colorbar = False)
        ax_cm.grid(False)
        ax_cm.set_title(
            f'{title_prefix}Confusion Matrix (threshold = {threshold:.2f})',
            fontsize = 14,
            fontweight = 'bold'
        )

        plt.tight_layout()
        plt.show()

        return report_df, metrics_df

    except Exception as e:
        raise RuntimeError(
            f'Failure to generate ROC curves, PR, confusion matrix and classification report: {e}'
        ) from e
