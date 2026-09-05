
# for each year, all prev uears are used for training
validation_years = [ 
    2019,
    2020,
    2021,
    2022
]

def create_temporal_folds(train_data, validation_years):
    """
    Create temporal folds for cross-validation.
    
    Parameters:
    train_data:
        DataFrame containing only the model-development
        training period.

    validation_years:
        Years that should act as temporal validation periods.
    
    Returns:
     folds:
        List of dictionaries.
        Each dictionary contains:

        - fold number
        - validation year
        - fold training data
        - fold validation data
        
    Example:

    Fold 1:
        train = 2015-2018
        validation = 2019

    Fold 2:
        train = 2015-2019
        validation = 2020

    and so on.
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

        assert len(fold_train) > 0
        assert len(fold_validation) > 0

        assert(
            fold_train["Date"].max() < fold_validation["Date"].min()
        )

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
        
    