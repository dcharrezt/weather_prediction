import pandas as pd
import numpy as np

aqp_path = "datasets/AQP_temperatures/AQP_temperatures.csv"

def interpret(line):
	# id,time(date hh:mm:ss),wind,temp,dew
	params = line.rstrip().split(',')
	mId = int(params[0])
	mTimestamp = str(params[1])
	mWind= float(params[2])
	mTemp= float(params[3])
	mDew= float(params[3])

	print(mTimestamp)
	yy=int(mTimestamp[:4])
	mo=int(mTimestamp[4:6])
	dd=int(mTimestamp[6:8])
	hh=int(mTimestamp[8:10])
	mm=int(mTimestamp[10:12])
	
	record={'id':mId, \
			'dt':{'yy':yy,'mo':mo,'dd':dd,'hh':hh,'mm':mm,'ss':00}, \
			'wind':mWind, \
			'temp':mTemp, \
			'dew':mDew}
	return record
	
def to_timestamp(line):

	line = str(line)
	yy=int(line[:4])
	mo=int(line[4:6])
	dd=int(line[6:8])
	hh=int(line[8:10])
	mm=int(line[10:12])
	
	record = str(yy)+'-'+str(mo)+'-'+str(dd)+' '+str(hh)+':'+str(mm)+':00'
	return record

def farenheit_to_celcius(ms):
	tmp = []
	for i in ms:
		tmp.append(5*(i-32)/9)
	return tmp

def mph_to_kph(ms):
	tmp = []
	for i in ms:
		tmp.append(i*1.60934)
	return tmp

def preprocess_aqp_dataset():

	feature_list =["station_id","date_time","wind_speed","temperature","dew_point"]
	df = pd.read_csv(aqp_path, names = feature_list)
	
	df = df.interpolate('linear')
	
	tmp = []
	timestamp = np.array(df['date_time'])
	for i in timestamp:
		tmp.append(to_timestamp(i))
	df['date_time'] = pd.to_datetime(tmp)
	
	#To celcius from farenheit
	celcius = farenheit_to_celcius(np.array(df['temperature']))
	df['temperature'] = celcius
	
	celcius_dew = farenheit_to_celcius(np.array(df['dew_point']))
	df['dew_point'] = celcius_dew
	
	#MPH to KPH
	kph = mph_to_kph(np.array(df['wind_speed']))
	df['wind_speed'] = kph
	
	return df	

#df = preprocess_aqp_dataset()

#print(df.dtypes)
#print(df.head(10))
#print(len(df))






