#테스트 데이터 트레인 데이터 나누기

import pandas as pd
from sklearn.model_selection import train_test_split

def split_data(input_file, train_file, test_file, test_size=0.2, random_state=42):
    # Load the dataset
    data = pd.read_csv(input_file)

    # Split the data into training and testing sets
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)

    # Save the split datasets
    train_data.to_csv(train_file, index=False)
    test_data.to_csv(test_file, index=False)

    print(f"Training data saved to {train_file}")
    print(f"Testing data saved to {test_file}")

if __name__ == "__main__":
    # File paths
    input_file = "c:/Users/user/Desktop/BigDataStudy/data/processed_data/ImportantFactors_no_outliers.csv"
    train_file = "c:/Users/user/Desktop/BigDataStudy/data/processed_data/TrainFactors.csv"
    test_file = "c:/Users/user/Desktop/BigDataStudy/data/processed_data/TestFactors.csv"

    # Split the data
    split_data(input_file, train_file, test_file)