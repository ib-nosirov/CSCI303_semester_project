import numpy as np
import pandas as pd
import calendar
import time
import sklearn
from sklearn.preprocessing import MinMaxScaler


class Process():
    def process(self, pd_obj):
        self.convert_timestamps(pd_obj)
        pd_obj.fillna(-1.0, inplace=True)
        pd_obj = self.drop(pd_obj)
        np_obj = pd_obj.to_numpy()
        return self.create_timeseries(np_obj)

# Pandas helper methods
    def convert_timestamps(self, pd_obj):
        pd_obj.loc[:, 'RE_DATE'] = self.to_seconds('RE_DATE', pd_obj)
        self.get_hospital_time(pd_obj)

    def get_hospital_time(self, pd_obj):
        hospital_time = self.to_seconds('Discharge time', pd_obj)
        - self.to_seconds('Admission time', pd_obj)
        pd_obj['Hospital time'] = hospital_time

    def to_seconds(self, column, pd_obj):
        return (pd_obj.loc[:, column]
                - pd.Timestamp("1970-01-01"))//pd.Timedelta('1s')

    def drop(self, pd_obj):
        pd_obj = pd_obj.drop(['Discharge time', 
            'Admission time'], axis=1)
        return pd_obj

# Numpy helper methods
    def create_timeseries(self, np_obj):
        arr = [np.empty([2, 80])]
        for i in range(1, 376):
            arr = np.insert(arr, i, self.make_patient(i, np_obj), axis=0)
        arr = arr[1:,:,:]
        np.random.shuffle(arr)
        return arr

    def make_patient(self, patient_num, np_obj):
        i = self.get_patient_index(patient_num, np_obj)
        arr = [np_obj[i, :]]
        while ((i < 6120) and (np_obj[i, 0] != patient_num+1)):
            i = i+1
        arr = np.append(arr, [np_obj[i-1, :]], axis=0)
        return arr

    def get_patient_index(self, patient_num, np_obj):
        for i in range(6120):
            if (patient_num == np_obj[i, 0]):
                return i

# Train-test-split
    def train_test_split(self, np_obj):
        Ytrain = np_obj[:300,0,4:5].flatten()
        Ytrain = Ytrain.reshape((Ytrain.shape[0],1))
        Ytest = np_obj[300:,0,4:5].flatten()
        Ytest = Ytest.reshape((Ytest.shape[0],1))
        np_obj = np.delete(np_obj,4,axis=2)
        Xtrain = np_obj[:300,:,:]
        Xtest = np_obj[300:,:,:]
        return Xtrain, Ytrain, Xtest, Ytest

    def scale(self, np_obj):
        scaler = sklearn.preprocessing.MinMaxScaler()
        for i in range(len(np_obj)):
            np_obj[i,:,:] = scaler.fit_transform(np_obj[i,:,:])
        return np_obj

