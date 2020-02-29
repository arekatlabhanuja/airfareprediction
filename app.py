import shutil
import random
import datetime
from flask import Flask
from flask import request, redirect
from flask import render_template
import mysql.connector
import datetime

import pandas as pd
import numpy as np 
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.svm import SVC

from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


app = Flask(__name__)
 
@app.route("/")
def index():
	return render_template('index.html')

@app.route("/about")
def about():
	return render_template('about.html')

@app.route("/contact")
def contact():
	return render_template('contact.html')

@app.route("/register", methods=['GET', 'POST'])
def register():
	if request.method == 'GET':
		return render_template('register.html')
	else:
		email = request.form['email']
		pwd = request.form['pwd']
		name = request.form['name']
		addr = request.form['addr']
		cnx = mysql.connector.connect(user='root', database='airfare')
		cursor = cnx.cursor()
		add_user = ("INSERT INTO USERS "
					   "(email, password, fullname, address) "
					   "VALUES (%s, %s, %s, %s)")
		data_user = (email, pwd, name, addr)
		cursor.execute(add_user, data_user)
		cnx.commit()
		cursor.close()
		cnx.close()
		return render_template('login.html')

@app.route("/adminlogin", methods=['GET', 'POST'])
def adminlogin():
	if request.method == 'GET':
		return render_template('adminlogin.html')
	else:
		email = request.form['email']
		pwd = request.form['pwd']	
		if email=='admin' and pwd=='secret':
			return redirect('process')
		else:
			return render_template('adminlogin.html')

@app.route("/searchresult2", methods=['GET', 'POST'])
def searchresult2():
	fromtxt = request.form['from']
	totxt = request.form['to']
	budtxt = request.form['bud']
	dttxt = request.form['dt']
	dt=datetime.datetime.strptime(dttxt, "%Y-%m-%d").date()
	m=dt.month
	d=dt.weekday()
	citydata = pd.read_csv('Data_Set - VZG-'+totxt+'.csv')
	citydata = citydata[citydata['Fare']<(int(budtxt))] # & citydata['Fare']<(int(budtxt)+100)]
	# pd.to_datetime(citydata['Date'], format='%m/%d/%Y')
	try:
		models.predict(citydata)
	except:
		pass
	citydata = citydata[pd.to_datetime(citydata['Date'], format='%m/%d/%Y').dt.dayofweek==d] # & citydata['Fare']<(int(budtxt)+100)]
	#dt = pd.datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
	#citydata['Date']= pd.to_datetime(citydata['Date']).to_frame()+ datetime.timedelta(days=30) #pd.to_datetime( citydata['Date']) - dt)
	for i, row in citydata.iterrows():
		citydata.set_value(i,'Date',dt+datetime.timedelta(days=random.randint(1,3*i)))
	citydata['Date'] = pd.to_datetime(citydata['Date'], format='%Y-%m-%d')
	#citydata['Date']= pd.to_datetime(citydata['Date']).to_frame()+ datetime.timedelta(days=pd.to_datetime( citydata['Date']) - dt)
	citydata=citydata.sort_values(['Fare'], ascending=[True])
	#citydata=citydata[1:3,]
	series = citydata.loc[citydata['Fare'].idxmin()]
	#citydatabest = series.iloc[[0,1,3]].to_frame()
	citydatabest = series.to_frame()
	#citydatabest = citydatabest['Source','Destination', 'Fare']
	# .dt.dayofweek
	Data = {'Source':  [fromtxt],
        'Destination': [totxt], 
		'Budget': [budtxt], 
		'Date': [dttxt]
        }	
	df = pd.DataFrame (Data, columns = ['Source','Destination','Budget', 'Date'])
	return render_template('searchresult2.html',
		tables=[
		#citydatabest.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'),
		df.to_html(classes='data table  table-responsive table-bordered table-striped table-inverse'),
		citydata.to_html(classes='data table  table-responsive table-bordered table-striped table-inverse')
		], 
		titles=citydata.columns.values)

		
@app.route("/login", methods=['GET', 'POST'])
def login():
	if request.method == 'GET':
		return render_template('login.html')
	else:
		email = request.form['email']
		pwd = request.form['pwd']
		cnx = mysql.connector.connect(user='root', database='airfare')
		cursor = cnx.cursor()
		add_user = ("SELECT COUNT(*) FROM USERS WHERE email=%s and password=%s")
		data_user = (email, pwd)
		cursor.execute(add_user, data_user)
		k=1
		for row in cursor:
			k=row[0]
		cursor.close()
		cnx.close()
		if k==1:
			print('in')
			return render_template('index.html')
		else:
			print('out')
			return render_template('login.html')


@app.route("/prediction")
def prediction():
	return render_template('prediction.html')
	
@app.route("/start")
def start():
	shuffled_data = pd.read_csv('Clean_shuffled_data_final.csv')

	shuffled_data1 = pd.read_csv('Data_Set - VZG-BOM.csv')
	shuffled_data2 = pd.read_csv('Data_Set - VZG-CHN.csv')
	shuffled_data3 = pd.read_csv('Data_Set - VZG-COH.csv')
	shuffled_data4 = pd.read_csv('Data_Set - VZG-DLH.csv')
	shuffled_data5 = pd.read_csv('Data_Set - VZG-HYD.csv')
	shuffled_data6 = pd.read_csv('Data_Set - VZG-KOL.csv')
	shuffled_data7 = pd.read_csv('Data_Set - VZG-PNQ.csv')
	shuffled_data8 = pd.read_csv('Data_Set - VZG-VJY.csv')
	# In[5]:
	# features considered, 
	# Airways, 
	# Arrival Time, Departure Time
	# Flight Duration
	# Days to booking and Flight
	# Hopping
	#print (shuffled_data.head())
	# In[6]:
	Flight_Price = shuffled_data['Total_Fare']
	# In[7]:
	del shuffled_data['Total_Fare']
	# In[8]:
	Flight_Price1 = shuffled_data1['Fare']
	# In[7]:
	del shuffled_data1['Fare']
	Flight_Price2 = shuffled_data2['Fare']
	# In[7]:
	del shuffled_data2['Fare']
	Flight_Price3 = shuffled_data3['Fare']
	# In[7]:
	del shuffled_data3['Fare']
	Flight_Price4 = shuffled_data4['Fare']
	# In[7]:
	del shuffled_data4['Fare']
	Flight_Price5 = shuffled_data5['Fare']
	# In[7]:
	del shuffled_data5['Fare']
	Flight_Price6 = shuffled_data6['Fare']
	# In[7]:
	del shuffled_data6['Fare']
	Flight_Price7 = shuffled_data7['Fare']
	# In[7]:
	del shuffled_data7['Fare']
	Flight_Price8 = shuffled_data8['Fare']
	# In[7]:
	del shuffled_data8['Fare']
	from sklearn.model_selection import train_test_split
	# In[9]:
	shuffled_data_train, shuffled_data_test, Flight_Price_train, Flight_Price_test = train_test_split(shuffled_data,Flight_Price, test_size=0.05, random_state=42)
	try:
		shuffled_data_train1, shuffled_data_test1, Flight_Price_train1, Flight_Price_test1 = train_test_split(shuffled_data1,Flight_Price1, test_size=0.05, random_state=42)
		shuffled_data_train2, shuffled_data_test2, Flight_Price_train2, Flight_Price_test2 = train_test_split(shuffled_data2,Flight_Price2, test_size=0.05, random_state=42)
		shuffled_data_train3, shuffled_data_test3, Flight_Price_train3, Flight_Price_test3 = train_test_split(shuffled_data3,Flight_Price3, test_size=0.05, random_state=42)
		shuffled_data_train4, shuffled_data_test4, Flight_Price_train4, Flight_Price_test4 = train_test_split(shuffled_data4,Flight_Price4, test_size=0.05, random_state=42)
		shuffled_data_train5, shuffled_data_test5, Flight_Price_train5, Flight_Price_test5 = train_test_split(shuffled_data5,Flight_Price5, test_size=0.05, random_state=42)
		shuffled_data_train6, shuffled_data_test6, Flight_Price_train6, Flight_Price_test6 = train_test_split(shuffled_data6,Flight_Price6, test_size=0.05, random_state=42)
		shuffled_data_train7, shuffled_data_test7, Flight_Price_train7, Flight_Price_test7 = train_test_split(shuffled_data7,Flight_Price7, test_size=0.05, random_state=42)
		shuffled_data_train8, shuffled_data_test8, Flight_Price_train8, Flight_Price_test8 = train_test_split(shuffled_data8,Flight_Price8, test_size=0.05, random_state=42)
	except:
		pass
		
	return render_template('start.html', 
		tables=[shuffled_data_train.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'), 
		shuffled_data_test.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse')],
		tables2=[shuffled_data_train1.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'), 
		shuffled_data_test1.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'),
		shuffled_data_train2.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'), 
		shuffled_data_test2.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'),
		shuffled_data_train3.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'), 
		shuffled_data_test3.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'),
		shuffled_data_train4.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'), 
		shuffled_data_test4.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'),
		shuffled_data_train5.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'), 
		shuffled_data_test5.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'),
		shuffled_data_train6.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'), 
		shuffled_data_test6.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'),
		shuffled_data_train7.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'), 
		shuffled_data_test7.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'),
		shuffled_data_train8.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'), 
		shuffled_data_test8.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse')], 
		fptr=Flight_Price_train, 
		fpte=Flight_Price_test,
		titles=[shuffled_data_train.columns.values,shuffled_data_test.columns.values],
		titles2=[shuffled_data_train1.columns.values,shuffled_data_test1.columns.values]
		)	

@app.route("/train")
def train():
	shuffled_data = pd.read_csv('Clean_shuffled_data_final.csv')
	# In[5]:
	# features considered, 
	# Airways, 
	# Arrival Time, Departure Time
	# Flight Duration
	# Days to booking and Flight
	# Hopping
	#print (shuffled_data.head())
	# In[6]:
	Flight_Price = shuffled_data['Total_Fare']
	# In[7]:
	del shuffled_data['Total_Fare']
	# In[8]:
	from sklearn.model_selection import train_test_split
	# In[9]:
	shuffled_data_train, shuffled_data_test, Flight_Price_train, Flight_Price_test = train_test_split(shuffled_data,Flight_Price, test_size=0.05, random_state=42)


	# In[11]:

	models = [LinearRegression(),
				  RandomForestRegressor(n_estimators=100, max_features='sqrt'),
				  KNeighborsRegressor(n_neighbors=6),
				  SVR(kernel='linear'),
				  LogisticRegression()
				  ]
	TestModels = pd.DataFrame()
	tmp = {}


	# In[12]:

	model_name_list = list()
	r2_score_list = list()
	mean_abs_err_list = list()
	mean_sq_err_list = list()
	i = 0
	for model in models:
		m = str(model)
		tmp['Model'] = m[:m.index('(')]
		model.fit(shuffled_data_train, Flight_Price_train)
		predicted_value = model.predict(shuffled_data_test)
		indi_r2_score = r2_score(Flight_Price_test, predicted_value )
		r2_score_list.insert(i,indi_r2_score)
	#     model_name_list.insert(i,m)
	#     print confusion_matrix(Flight_Price_test,predicted_value)
		mae =  mean_absolute_error(Flight_Price_test,predicted_value)
		mse = mean_squared_error(Flight_Price_test,predicted_value)
		mean_abs_err_list.insert(i,mae)
		mean_sq_err_list.insert(i,mse)
		i += 1


	# In[13]:

	model_name = ('LinearRegression','RandomForestRegressor','KNeighborsRegressor','SVR','LogisticRegression')


	# ##### The r2_score function computes R2, the coefficient of determination. It provides a measure of how well future samples are likely to be predicted by the model. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0

	# In[14]:

	# Plotting R2 score values in diff Supervised Regression Algorithms

	y_pos = np.arange(len(model_name))
	plt.figure(figsize=(15,5))
	bar1 = plt.bar(y_pos, r2_score_list, align='center', alpha=0.7)

	for i,rect in enumerate(bar1):
		height = rect.get_height()
		plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.3f' % float(r2_score_list[i]), ha='center', va='bottom')

	plt.xticks(y_pos, model_name)
	plt.ylabel('r2_score')
	plt.title('Comparison of R2_Score in diff Supervised ML Algorithms')
	plt.savefig('r2_score_vs_algo.png')
	shutil.copyfile('r2_score_vs_algo.png', 'static/images/r2_score_vs_algo.png')  
	#plt.show()
	


	# In[15]:

	y_pos = np.arange(len(model_name))
	plt.figure(figsize=(15,5))
	bar1 = plt.bar(y_pos, mean_abs_err_list, align='center', alpha=0.7)

	for i,rect in enumerate(bar1):
		height = rect.get_height()
		plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.3f' % float(mean_abs_err_list[i]), ha='center', va='bottom')

	plt.xticks(y_pos, model_name)
	plt.ylabel('mean abs error')
	plt.title('Comparison of Mean Absolute Error in diff Supervised ML Algorithms')
	plt.savefig('MAE_vs_algo.png')
	shutil.copyfile('MAE_vs_algo.png', 'static/images/MAE_vs_algo.png')  
	#plt.show()
	


	# In[16]:

	#y_pos = np.arange(len(model_name))
	#plt.figure(figsize=(15,5))
	#bar1 = plt.bar(y_pos, mean_sq_err_list, align='center', alpha=0.7)

	#for i,rect in enumerate(bar1):
	#	height = rect.get_height()
	#	plt.text(rect.get_x() + rect.get_width()/2.0, height, '%.3f' % float(mean_sq_err_list[i]), ha='center', va='bottom')

	#plt.xticks(y_pos, model_name)
	#plt.ylabel('mean square error')
	#plt.title('Comparison of Mean Square Error in diff Supervised ML Algorithms')
	#plt.savefig('MSE_vs_algo.png')
	#shutil.copyfile('MAE_vs_algo.png', 'static/images/MAE_vs_algo.png')  
	##plt.show()

	return render_template('train.html')
		
@app.route("/process")
def process():
	dtVerticalScrollExample = pd.read_csv('Clean_shuffled_data_final.csv')
	dtVerticalScrollExample1 = pd.read_csv('Data_Set - VZG-BOM.csv')
	dtVerticalScrollExample2 = pd.read_csv('Data_Set - VZG-CHN.csv')
	dtVerticalScrollExample3 = pd.read_csv('Data_Set - VZG-COH.csv')
	dtVerticalScrollExample4 = pd.read_csv('Data_Set - VZG-DLH.csv')
	dtVerticalScrollExample5 = pd.read_csv('Data_Set - VZG-HYD.csv')
	dtVerticalScrollExample6 = pd.read_csv('Data_Set - VZG-KOL.csv')
	dtVerticalScrollExample7 = pd.read_csv('Data_Set - VZG-PNQ.csv')
	dtVerticalScrollExample8 = pd.read_csv('Data_Set - VZG-VJY.csv')
	#print(dtVerticalScrollExample)
	return render_template('process.html',
		tables=[dtVerticalScrollExample.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse')], 
		tables2=[dtVerticalScrollExample1.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'),
		dtVerticalScrollExample2.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'),
		dtVerticalScrollExample3.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'),
		dtVerticalScrollExample4.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'),
		dtVerticalScrollExample5.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'),
		dtVerticalScrollExample6.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'),
		dtVerticalScrollExample7.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'),
		dtVerticalScrollExample8.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse')], 
		titles=dtVerticalScrollExample.columns.values,
		titles2=dtVerticalScrollExample1.columns.values)

@app.route("/rfg")
def rfg():
	shuffled_data = pd.read_csv('Clean_shuffled_data_final.csv')


	# In[5]:

	# features considered, 
	# Airways, 
	# Arrival Time, Departure Time
	# Flight Duration
	# Days to booking and Flight
	# Hopping

	print (shuffled_data.head())


	# In[6]:

	Flight_Price = shuffled_data['Total_Fare']


	# In[7]:

	del shuffled_data['Total_Fare']


	# In[8]:

	from sklearn.model_selection import train_test_split


	# In[9]:

	shuffled_data_train, shuffled_data_test, Flight_Price_train, Flight_Price_test = train_test_split(shuffled_data,Flight_Price, test_size=0.05, random_state=42)

	RF_model = RandomForestRegressor(n_estimators=100, max_features='sqrt')


	# In[18]:

	RF_model = RF_model.fit(shuffled_data_train, Flight_Price_train)


	# In[19]:

	RF_model_pred_flight_price = RF_model.predict(shuffled_data_test)

	plt.figure(figsize=(15,5))
	plt.scatter(RF_model_pred_flight_price, Flight_Price_test)
	plt.xlabel('Prediction')
	plt.ylabel('Real Value')

	diagonal = np.linspace(0, np.max(Flight_Price_test), 100)
	plt.plot(diagonal, diagonal, '-r')
	plt.savefig('rfg_pred_real.png')
	shutil.copyfile('rfg_pred_real.png', 'static/images/rfg_pred_real.png')  
	#plt.show()
	#print(dtVerticalScrollExample)
	return render_template('rfg.html')

		
if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000)