import pandas as pd

from sklearn.preprocessing import OneHotEncoder

# Define the features to be used in the model
numeric_features = [
    "Player1Rank",
    "Player2Rank",
    "Player1Points",
    "Player2Points",

    "Player1MatchesBefore",
    "Player2MatchesBefore",
    "ExperienceDifference",

    "Player1WinRateBefore",
    "Player2WinRateBefore",
    "WinRateDifference",

    "Player1Recent5WinRate",
    "Player2Recent5WinRate",
    "Recent5WinRateDifference",

    "Player1SurfaceMatchesBefore",
    "Player2SurfaceMatchesBefore",
    "SurfaceMatchesDifference",

    "Player1SurfaceWinRateBefore",
    "Player2SurfaceWinRateBefore",
    "SurfaceWinRateDifference",

    "H2HMatchesBefore",
    "Player1H2HWinRateBefore",
    "Player2H2HWinRateBefore",
    "DifferenceH2HWinRateBefore",

    "Player1DaysSinceLastMatch",
    "Player2DaysSinceLastMatch",
    "DaysSinceLastMatchDifference",

    "MissingPlayer1DaysSinceLastMatch",
    "MissingPlayer2DaysSinceLastMatch",

    "RankDifference",
    "PointsDifference"
]


categorical_features = [
    "Series",
    "Court",
    "Surface",
    "Round",
    "Best of"
]

def fit_preprocessor(training_data):
    
    # Calculate the median for each numeric feature in the training data
    numeric_medians= training_data[numeric_features].median()
    
    # Calculate the mode for each categorical feature in the training data
    categorical_modes = {}

    for column in categorical_features:
        training_mode = training_data[column].mode().iloc[0]
        categorical_modes[column] = training_mode
   
    # Create a copy of the categorical features from the training data, we don't want to modify the original training data 
    categorical_training = training_data[categorical_features].copy()
    
    # Fill missing values in the categorical features with the mode of each column
    for column in categorical_features:
        categorical_training[column] = (categorical_training[column].fillna(categorical_modes[column]))
    
    # Create a OneHotEncoder instance to encode the categorical features
    #sparse_output=False ensures that the output is a dense array instead of a sparse matrix
    #handle_unknown="ignore" ensures that if a category is encountered in the validation or test set that was not present in the training set, it will be ignored instead of raising an error
    one_hot_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    one_hot_encoder.fit(
       categorical_training
    )
    
    return(
        numeric_medians,
        categorical_modes,
        one_hot_encoder
    )
        
def transform_data(data, numeric_medians, categorical_modes, one_hot_encoder):
    
    numeric_data = (data[numeric_features].copy())

    # Fill missing values in the numeric features with the median of each column
    numeric_data = numeric_data.fillna(numeric_medians)
    
    
    categorical_data = ( data[categorical_features].copy())
    # Fill missing values in the categorical features with the mode of each column
    for column in categorical_features:
        categorical_data[column] = categorical_data[column].fillna(categorical_modes[column])
    
    # Transform the categorical features using the fitted OneHotEncoder
    categorical_encoded = one_hot_encoder.transform(categorical_data)
    
    # Get appropriate column names
    encoded_feature_names = (
        one_hot_encoder.get_feature_names_out(
            categorical_features
        )
    )
    
    #Create a DataFrame for the encoded categorical features with appropriate column names
    #index=data.index ensures that the index of the new DataFrame matches the index of the original data
    encoded_categorical_df = pd.DataFrame(
        categorical_encoded,
        columns=encoded_feature_names,
        index=data.index
    )
    
    # Concatenate the numeric features and the encoded categorical features into a single DataFrame
    #axis=1 ensures that the concatenation is done column-wise
    transformed_data = pd.concat(
        [numeric_data, encoded_categorical_df],
        axis=1
    )
    
    return transformed_data