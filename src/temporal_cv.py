# in this file we create temporal cross-validation folds where the model is always trained on the past and validated on the future

# we perform four validation experiments
# for validation year, all previous years are used for training
validation_years = [ 
    2019,
    2020,
    2021,
    2022
]

# we create a function to construct/check and report the temporal fold structure
def create_temporal_folds(train_data, validation_years):
    """
    Create temporal cross-validation folds.

    Parameters:
        train_data (DataFrame): Model-development data.
        validation_years (list): Years used as validation periods.

    Returns:
        list: Summary information for each temporal fold.

    Each fold uses all matches before the validation year for training
    and matches from that year for validation.

    Example:
        Fold 1: train = 2015–2018, validation = 2019
        Fold 2: train = 2015–2019, validation = 2020
        ...
    """
    # list to store different cross-validation folds
    temporal_cv_rows = []

    # each validation_year has a fold: 2019 -> fold 1
    for fold_number, validation_year in enumerate(
        validation_years, 
        start=1
    ):

        # create training part for each fold: all matches before the validation year
        fold_train = train_data.loc[
            train_data["Date"].dt.year < validation_year
        ].copy()

        # create validation part for each fold: all matches in the validation year
        fold_validation = train_data.loc[
            train_data["Date"].dt.year == validation_year
        ].copy()

        # we perform basic safety checks
        assert len(fold_train) > 0
        assert len(fold_validation) > 0
        assert(fold_train["Date"].max() < fold_validation["Date"].min())

        temporal_cv_rows.append(
            {
                "Fold": fold_number,
                "TrainingMatches": len(fold_train),
                "ValidationMatches": len(fold_validation),
                "TrainingStartYear": fold_train["Date"].dt.year.min(),
                "TrainingEndYear": fold_train["Date"].dt.year.max(),
                "ValidationYear": validation_year
            }
        )
        
    return temporal_cv_rows
        
    