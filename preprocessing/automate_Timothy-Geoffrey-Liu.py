import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

RAW_DATA_PATH = 'student_performance_prediction_raw.csv'
PREPROCESSED_DATA_PATH = 'preprocessing/student_performance_prediction_preprocessing.csv'
NUMERIC_COLUMNS = ['Study Hours per Week', 'Attendance Rate', 'Previous Grades']

def load_dataset(dataset_path):
    return pd.read_csv(dataset_path)

def basic_cleaing(df):
    df = df.drop(columns = ['Student ID'])
    df = df.dropna()
    df = df.drop_duplicates()

    return df

def encode_features(df):
    df['Passed'] = df['Passed'].map({'No': 0, 'Yes': 1})
    df['Participation in Extracurricular Activities'] = df['Participation in Extracurricular Activities'].map({'No': 0, 'Yes': 1})
    df = pd.get_dummies(df, columns = ['Parent Education Level'], drop_first = True)

    return df

def remove_invalid_values(df):
    df = df[
        (df['Study Hours per Week'] >= 0) &
        (df['Attendance Rate'].between(0, 100)) &
        (df['Previous Grades'].between(0, 100))
    ]

    return df

def remove_outliers(df, columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    return df

def scale_numeric_features(df, columns):
    standard_scaler = StandardScaler()
    df[columns] = standard_scaler.fit_transform(df[columns])

    return df

def separate_features_and_target(df):
    X = df.drop(columns = ['Passed'])
    y = df['Passed']

    return X, y

def main():
    df = load_dataset(RAW_DATA_PATH)
    df = basic_cleaing(df)
    df = encode_features(df)
    df = remove_invalid_values(df)
    df = remove_outliers(df, NUMERIC_COLUMNS)
    df = scale_numeric_features(df, NUMERIC_COLUMNS)
    X, y = separate_features_and_target(df)

    df.to_csv(PREPROCESSED_DATA_PATH, index = False)
    print(f"Preprocessing completed. Output saved to: ./{PREPROCESSED_DATA_PATH}")

if __name__ == '__main__':
    main()
