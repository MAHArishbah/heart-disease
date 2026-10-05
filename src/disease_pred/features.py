"""The fitted preprocessor.Everything in here learns a parameter from the training data -- imputer
medians, the scaler's mean/sd, the encoder's category list, which columns had
any missingness at all. That is exactly why it must live *inside* the Pipeline
rather than being reapplied by hand at predict time: recomputing any of it on
incoming rows is train/serve skew."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def build_preprocessor(numeric, binary, categorical,indicators) -> ColumnTransformer:
    """ColumnTransformer, with: remainder='drop'.while serving passthrough silently appends any unexpected column in an incoming
    payload to the design matrix and breaks the coefficient alignment. 'drop'
    fails loudly instead. serve.py also reindexes to the saved column list, so
    the guard is doubled.The MissingIndicator block is deliberate .
    """
    num_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scalar", StandardScaler()),
    ])
    binary_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
    ])
    nominal_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("ohe", OneHotEncoder(drop="first",
                              handle_unknown="infrequent_if_exist",
                              sparse_output=False)),
    ])
    blocks=  [
                ("num", num_pipeline, numeric),
                ("bin", binary_pipeline, binary),
                ("nom", nominal_pipeline, categorical),
             ]
    if indicators:
        blocks.append(("missing", MissingIndicator(features="all"), indicators))

    return ColumnTransformer(
      blocks,remainder="drop",
    )


def output_names(preprocessor: ColumnTransformer) -> list[str]:
    """`feature_names`: strip the ColumnTransformer's block prefix."""
    return [n.split("__", 1)[1] for n in preprocessor.get_feature_names_out()]
