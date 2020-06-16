# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 17:44:24 2020

@author: caiom
"""

import serial
# import csv
import pandas
import time
import matplotlib.pyplot as plt

ser = serial.Serial('COM3', baudrate=230400)
ser.flushInput()

data_sets = 100
t = 'speed1_3s_'
csv_text = '.csv'
while data_sets:
    file_name = t + str(data_sets) + csv_text
    print(file_name)
    data_points = 6660*3
    data_stream = []
    print(time.time())
    while data_points:
        ser_bytes = ser.readline()
        decoded_bytes = float(ser_bytes[0:len(ser_bytes)-3].decode("utf-8"))
        data_stream.append(decoded_bytes)
        data_points = data_points - 1
    print(time.time())
    df = pandas.DataFrame(data={"accZ": data_stream})
    df.to_csv(file_name, sep=',', index=False)
#    with open(file_name,"a") as f:
#        writer = csv.writer(f,delimiter=",")
#        writer.writerow(data_stream)
    data_sets = data_sets - 1
ser.close()
plt.figure()
df.plot(figsize=(300, 6))
