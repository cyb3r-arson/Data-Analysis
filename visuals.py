import pandas as pd
import matplotlib.pyplot as plt

# Task 1: Load and Explore the Dataset
try:
    # Load the dataset 
    df = pd.read_csv('iris.csv')  

    # Display the first few rows to inspect the data
    print("First few rows of the dataset:")
    print(df.head())

    # Display the data types of columns
    print("\nData types of columns:")
    print(df.dtypes)

    # Check for missing values
    print("\nMissing values in the dataset:")
    print(df.isnull().sum())

    # Fill missing values with the mean 
    df.fillna(df.mean(), inplace=True)  
    
    # Task 2: Basic Data Analysis
    # Basic statistical analysis of numerical columns
    print("\nBasic Statistics of Numerical Columns:")
    print(df.describe())

    # Group by a categorical column (e.g., 'variety' if using Iris dataset) and compute mean for numeric columns only
    if 'variety' in df.columns:
        grouped = df.groupby('variety').mean(numeric_only=True)
        print("\nGrouped Data (Mean values per variety):")
        print(grouped)
    else:
        print("\nNo 'variety' column found for grouping.")

    # Task 3: Data Visualization

    # Line chart: Example of sales over time (replace 'Date' and 'Sales' with actual columns)
    if 'Date' in df.columns and 'Sales' in df.columns:
        plt.figure(figsize=(10, 6))
        plt.plot(df['Date'], df['Sales'])
        plt.title('Sales Trend Over Time')
        plt.xlabel('Date')
        plt.ylabel('Sales')
        plt.grid(True)
        plt.show()

    # Bar chart: Comparison of numerical values across categories (e.g., average petal length per variety)
    if 'variety' in df.columns and 'petal.length' in df.columns:
        plt.figure(figsize=(8, 5))
        df.groupby('variety')['petal.length'].mean().plot(kind='bar')
        plt.title('Average Petal Length per Variety')
        plt.xlabel('Variety')
        plt.ylabel('Average Petal Length')
        plt.show()

    # Histogram: Distribution of a numerical column (e.g., sepal length)
    if 'sepal.length' in df.columns:
        plt.figure(figsize=(8, 5))
        df['sepal.length'].hist(bins=20)
        plt.title('Distribution of Sepal Length')
        plt.xlabel('Sepal Length')
        plt.ylabel('Frequency')
        plt.show()

    # Scatter Plot: Relationship between two numerical columns (e.g., sepal length vs petal length)
    if 'sepal.length' in df.columns and 'petal.length' in df.columns:
        plt.figure(figsize=(8, 5))
        plt.scatter(df['sepal.length'], df['petal.length'])
        plt.title('Sepal Length vs Petal Length')
        plt.xlabel('Sepal Length')
        plt.ylabel('Petal Length')
        plt.show()

except FileNotFoundError:
    print("Error: The file was not found.")
except pd.errors.EmptyDataError:
    print("Error: No data found in the file.")
except Exception as e:
    print(f"An error occurred: {e}")
