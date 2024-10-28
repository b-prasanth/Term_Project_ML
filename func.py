import numpy as np
import pandas as pd
import math

#Function to calculate standard deviation.
def calc_std(df_col, df):
    mean = calc_mean(df_col, df)
    var = 0
    for row in df[df_col]:
        var += (row - mean) ** 2
        var = var / len(df)
        std = math.sqrt(var)
    return std

#Function to calculate mean
def calc_mean(df_col,df):
    sum=0
    for row in df[df_col]:
        sum+=row
    return sum/len(df)

#Function to standardize dataframe
def standardized(df):
    standardized_dataset=df
    ignor_cols=['ShelveLoc_Good', 'ShelveLoc_Medium', 'US_Yes', 'Urban_Yes']
    numeric_cols=standardized_dataset.select_dtypes(include=['int64',
    'float64','number']).columns
    for cols in numeric_cols:
        if cols not in ignor_cols:
            standardized_dataset[cols]=((standardized_dataset[cols]-
            calc_mean(cols,df))/calc_std(cols,df))
        else:
            continue
    return standardized_dataset

#Function to standardize target col of dataframe
def standardized_target(df):
    standardized_dataset=df
    mean=np.mean(standardized_dataset)
    std=np.std(standardized_dataset)
    standardized_dataset=((standardized_dataset-np.mean(df))/np.std(df))
    return standardized_dataset, mean, std

#Function to do reverse standardization
def inverse_std(df, mean, std):
    inverse_std_dataset=df
    inverse_std_dataset=(inverse_std_dataset*std)+mean
    return inverse_std_dataset

#Function to transpose a matrix
def matrix_transpose(matrix):
    matrix3= [[0 for x in range(len(matrix))] for y in range(len(matrix[0]))]
    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            matrix3[j][i]=(matrix[i][j])
    return matrix3

#Function for matrix multiplication
def matrix_multiply(matrix1, matrix2):
    m1_rows=len(matrix1)
    m1_cols=len(matrix1[0])
    m2_rows=len(matrix2)
    m2_cols=len(matrix2[0])
    # matrix3 = [[0] * m2_cols for _ in range(m1_rows)]
    matrix3=np.zeros((m1_rows,m2_cols))
    # print(np.shape(matrix3))
    for i in range(m1_rows):
        for j in range(m2_cols):
            for k in range(m1_cols):
                matrix3[i][j]+=(matrix1[i][k]*matrix2[k][j])
    return matrix3

#Function to generate a matrix of 1s
def matrix_of_ones(matrix):
    matrix3= [[1 for x in range(len(matrix))] for y in range(len(matrix))]
    # print(np.shape(matrix3))
    return matrix3

#Function to calculate median
def calc_median(df_col):
    median=0
    num=len(df_col)
    df_col.sort_values()
    if(num%2==0):
        temp1 = df_col.iloc[num // 2]
        temp2 = df_col.iloc[num // 2 - 1]
        median = (temp1 + temp2) / 2
        return median
    else:
        median = df_col.iloc[num // 2]
        return median

#Function to normalize dataframe
def normalized(data):
    normalized_data=data
    for cols in normalized_data:
        normalized_data[cols]=(normalized_data[cols]-min(normalized_data[cols]))/(max(normalized_data[cols])-min(normalized_data[cols]))
    return normalized_data

#Function to calculate geometric mean for a column
#Using Logarithmic and exponential calculation since the resultant of regular multiplication and division goes to infinity(overflow error)
def g_mean(df,cname):
    no_rows=len(df[cname])
    gmean=0.0
    for i in df[cname]:
        if i!=0:
            gmean = gmean + np.log(i)
        else:
            return 0.0
    gmean=gmean/no_rows
    geometric=float(np.exp(gmean))
    return round(geometric,2)

#Function to calculate harmonic mean manually.
def h_mean(df,cname):
    iterator = 0
    Hmean = 0
    for i in df[cname]:
        iterator = iterator + 1
        if i==0:
            return 0.0
        else:
            Hmean = Hmean + (1/i)
    Hmean = iterator/Hmean
    return round(Hmean,2)

#Function to calculate covariance matrix
def calc_cov(df):
    df_tmp_cols=df.columns
    df_tmp = df.to_numpy(dtype=np.float64)
    tmp2 = matrix_multiply(matrix_of_ones(df_tmp), df_tmp) / len(df_tmp)
    tmp_df3 = df_tmp - tmp2
    tmp_df4 = matrix_multiply(matrix_transpose(tmp_df3), tmp_df3) / ((len(df_tmp) - 1))
    cov_len=len(tmp_df4)
    for i in range(cov_len):
        cov_df=pd.DataFrame({df_tmp_cols[i]: tmp_df4[:,i]})
    # cov_df = pd.DataFrame(
    #     {'open': tmp_df4[:, 0], 'high': tmp_df4[:, 1], 'low': tmp_df4[:, 2], 'close': tmp_df4[:, 3],
    #      'volume': tmp_df4[:, 4]})
    cov_df.set_index([df_tmp_cols], inplace=True)
    return cov_df