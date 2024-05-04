from scipy import signal
from tsfresh import extract_features
import pandas as pd 
import numpy as np
from matplotlib import pyplot as plt

def convert_to_df(file_loc):
    file = pd.read_csv(file_loc)
    time_series = file.iloc[:,0]
    index_val = list(range(0,len(time_series),2))  
    time_series_50_hz = time_series.iloc[index_val]
    time_series_50_hz = time_series_50_hz.set_axis(range(0,len(time_series_50_hz)))
    time_index = list(range(0,len(time_series_50_hz),1)) 
    time_series_dict = {"Amp":list(time_series_50_hz) , "time": [i*1/50 for i in time_index]}
    time_series_df = pd.DataFrame(time_series_dict)
    return time_series_df

def visualize(time_series_df,x_label,y_label):
    plt.plot(time_series_df[x_label],time_series_df[y_label])
    plt.title(f"{y_label} Vs {x_label}")
    plt.xlabel(f"{x_label}")
    plt.ylabel(f"{y_label}")
    plt.show()

def fourier_transform(time_series_df):
    input_signal = time_series_df['Amp']
    sampling_rate = 50
    fft_data = np.fft.fft(input_signal)
    freqs = np.fft.fftfreq(n = len(fft_data), d = 1/sampling_rate)
    fig,ax = plt.subplots()
    ax.plot(freqs,abs(fft_data))
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Spectrum Magnitude')

def fourier_transform_range(time_series_df,lower_freq,upper_freq):
    input_signal = time_series_df['Amp']
    sampling_rate = 50
    fft_data = np.fft.fft(input_signal)
    freqs = np.fft.fftfreq(n = len(fft_data), d = 1/sampling_rate)
    fig,ax = plt.subplots()
    freq_mag_df = pd.DataFrame({'Spectral_Magnitude': abs(fft_data), 'Frequency': freqs})
    true_range = freq_mag_df[freq_mag_df['Frequency']>lower_freq][freq_mag_df['Frequency']<upper_freq]
    ax.plot(true_range['Frequency'],true_range['Spectral_Magnitude'])
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Spectrum Magnitude')

def band_pass_filter(time_series_df,lower_freq,upper_freq,sampling_rate):
    input_signal = time_series_df['Amp']
    #low pass filter
    fc = upper_freq
    w = fc/(sampling_rate/2) #normalisation by sampling_rate/2 for critical frequency
    b,a = signal.butter(5,w,'low',analog = False)    # N - order of bandpass filter  
    output = signal.filtfilt(b,a,input_signal)
    
    #high pass filter
    fch = lower_freq
    w = fc/(sampling_rate/2) 
    b,a = signal.butter(5,w,'high',analog = False)    # N - order of bandpass filter
    output = signal.filtfilt(b,a, output)
    
    filtered_df = pd.DataFrame({'fil_amp':output, 'time':time_series_df['time']})
    
    return filtered_df

def slice_dataframe(fil_df,lower_limit,upper_limit): #lower and upper time limits are used
    sliced_df = df[df['time']>lower_limit][df['time']<upper_limit]
    return sliced_df

def sta_lta_cal(slice_fil_df,LTA,STA):
    size = slice_fil_df['time'].max()
    sta_lta_df = pd.DataFrame(columns = ['Average','Time'])
    j = LTA  # j is used for time of reading in second
    while j<size:
        lta_mean = slice_fil_df['Amp'][slice_fil_df['time']>j-LTA][slice_fil_df['time']<j].abs().mean()
        sta_mean = slice_fil_df['Amp'][slice_fil_df['time']>j-STA][slice_fil_df['time']<j].abs().mean()
        time = j
        sta_lta_average = sta_mean/lta_mean
        sta_lta_df.loc[len(sta_lta_df)] = [sta_lta_average,time]
        j = j+5
    return sta_lta_df

def earthquake_noise_interval(sta_lta_df,threshold,interval): #threshold - 2.5 #interval = 1 sec
    ti = 0
    flag = 0
    li = []
    while ti<len(sta_lta_df):
        if sta_lta_df['Average'].iloc[ti] > threshold and flag == 0:
            li.append(sta_lta_df['Time'].iloc[ti])
            flag = 1
        elif sta_lta_df['Average'].iloc[ti]<threshold and flag == 1:
            li.append(sta_lta_df['Time'].iloc[ti])
            li.append('&')
            flag = 0
        ti = ti + interval
    return li

def feature_extraction(slice_fil_df,intervals,EQ_Nos):
    flag = 0
    for i in range(len(intervals)):
        #extract the time domain time series for intervals of earthquake
        if intervals[i] == '&' or intervals[i+1] == '&':
            continue
        a = intervals[i]
        b = intervals[i+1]
        extracted_interval = slice_fil_df[slice_fil_df['time']>a][slice_fil_df['time']<b]
        
        # add label to the data
        label = [EQ_Nos for x in extracted_interval['time']]
        extracted_interval['label'] = label

        # extract features of all the intervals
        j = extracted_interval['time'].min()
        while j<=extracted_interval['time'].max()-1:
            temp_interval = extracted_interval[extracted_interval['time']>j][extracted_interval['time']<j+1]
            feature = extract_features(timeseries_container = temp_interval, column_id = 'label', column_sort = 'time')
            if j == extracted_interval['time'].min() and flag == 0: #write to the external csv file
                feature.to_csv('earthquake_feature_EQ.csv',mode = 'a', header = True)
            else:
                feature.to_csv('earthquake_feature_EQ.csv',mode = 'a', header = False)
            j = j+1
        flag = 1

#for earthquake
import os
folder_to_view = '/kaggle/input/data-ascii/Event_ASCII'
for file in os.listdir(folder_to_view):
    try:
        loc = f"{folder_to_view}/{file}"
        print(loc)
        df = convert_to_df(loc)
        fil_df = band_pass_filter(df,0.1,10,50)
        slice_fil_df = slice_dataframe(fil_df,30,500)
        sta_lta_df = sta_lta_cal(slice_fil_df,60,2)
        intervals = earthquake_noise_interval(sta_lta_df,2.5,1)
        feature_extraction(slice_fil_df,intervals,'EQ')
    except:
        continue

#for noise

def feature_extraction(slice_fil_df,intervals,EQ_Nos):
    flag = 0
    for i in range(len(intervals)):
        #extract the time domain time series for intervals of earthquake
        if intervals[i] == '&' or intervals[i+1] == '&':
            continue
        a = intervals[i]
        b = intervals[i+1]
        extracted_interval = slice_fil_df[slice_fil_df['time']>a][slice_fil_df['time']<b]
        
        # add label to the data
        label = [EQ_Nos for x in extracted_interval['time']]
        extracted_interval['label'] = label

         # extract features of all the intervals
        j = extracted_interval['time'].min()
        while j<=extracted_interval['time'].max()-1:
            temp_interval = extracted_interval[extracted_interval['time']>j][extracted_interval['time']<j+1]
            feature = extract_features(timeseries_container = temp_interval, column_id = 'label', column_sort = 'time')
            if j == extracted_interval['time'].min() and flag == 0: #write to the external csv file
                feature.to_csv('earthquake_feature_Nos.csv',mode = 'a', header = True)
            else:
                feature.to_csv('earthquake_feature_Nos.csv',mode = 'a', header = False)
            j = j+1
        flag = 1

folder_to_view = '/kaggle/input/data-ascii/Noise_ASCII'
for file in os.listdir(folder_to_view):
    try:
        loc = f"{folder_to_view}/{file}"
        print(loc)
        df = convert_to_df(loc)
        fil_df = band_pass_filter(df,0.1,10,50)
        slice_fil_df = slice_dataframe(fil_df,30,500)
        sta_lta_df = sta_lta_cal(slice_fil_df,60,2)
        intervals = earthquake_noise_interval(sta_lta_df,2.5,1)
        feature_extraction(slice_fil_df,intervals,'Nos')
    except:
        continue





#saving extracted features
data_earthquake = pd.read_csv("/kaggle/working/earthquake_feature_EQ.csv")
data_noise = pd.read_csv('/kaggle/working/earthquake_feature_Nos.csv')
df_merged = pd.concat([data_earthquake,data_noise],ignore_index = True)
df_merged = df_merged.iloc[np.random.permutation(len(df_merged))]
df_merged = df_merged.reset_index()
df_merged.to_csv('Merged_features.csv')

#cleaning
df_merged = df_merged.drop('index', axis = 1)
df_merged.rename(columns = {'Unnamed: 0':'label'},inplace = True)
df_merged.fillna(0)

#converting string to float and non-numeric to 0
for col in df_merged:
    for i in range(len(df_merged.index)):
        try:
            df_merged[col].iloc[i] = float(df_merged[col].iloc[i])
        except:
            df_merged[col].iloc[i] = 0


#XGBoost model training
from sklearn.model_selection import train_test_split
from category_encoders.target_encoder import TargetEncoder


X = features.drop(column = 'Lables')
y = features['Labels']
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,stratify = y, random_state = 8)

from sklearn.pipeline import Pipeline
from category_encoders.target_encoder import TargetEncoder
from XGBoost import XGBClassifier

estimators = [
    ('encoder', TargetEncoder()),
    ('clf', XGBClassifier(random_state=8)) # can customize objective function with the objective paramenters
]
pipe = Pipeline(steps = estimators)

#hyperparameter search spaces
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
search_space = {
    'clf__max_depth': Integer(2,8),
    'clf__learning_rate': Real(0.001,1.0,prior = 'log-uniform'),
    'clf__subsample': Real(0.5,1.0),
    'clf__colsample_bytree': Real(0.5,1.0),
    'clf__colsample_bynode': Real(0.5,1.0),
    'clf__reg_alpha': Real(0.0,10.0),
    'clf__reg_lambda': Real(0.0,10.0),
    'clf__gamma': Real(0.0,10.0)
}

opt = BayesSearchCV(pip,search_space,cv=3,n_iter=10,scoring = 'roc_auc', random_state = 8)
opt.fit(X_train,y_train)

#Evaluate the model and make predictions
opt.best_estimator_
opt.best_score_
opt.score(X_test,y_test)
opt.predict(X_test)
opt.predict_proba(X_test)

