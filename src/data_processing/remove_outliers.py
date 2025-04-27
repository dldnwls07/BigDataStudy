#이상치 제거
import pandas as pd
import numpy as np

def remove_outliers(data, column_name, method='iqr'):
    if method == 'iqr':
        Q1 = data[column_name].quantile(0.25)
        Q3 = data[column_name].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_data = data[(data[column_name] >= lower_bound) & (data[column_name] <= upper_bound)]
    elif method == 'zscore':
        mean = data[column_name].mean()
        std = data[column_name].std()
        z_scores = (data[column_name] - mean) / std
        filtered_data = data[np.abs(z_scores) <= 3]
    else:
        raise ValueError("Unsupported method. Use 'iqr' or 'zscore'.")

    return filtered_data

if __name__ == "__main__":
    input_file = "c:/Users/user/Desktop/BigDataStudy/data/orginal_data/StudentPerformanceFactors.csv"
    output_file = "c:/Users/user/Desktop/BigDataStudy/data/orginal_data/StudentPerformanceFactors_no_outliers.csv"

    data = pd.read_csv(input_file)
    column_to_clean = 'Exam_Score'  
    cleaned_data = remove_outliers(data, column_to_clean, method='iqr')
    cleaned_data.to_csv(output_file, index=False)
    print(f"Outliers removed and cleaned data saved to {output_file}")