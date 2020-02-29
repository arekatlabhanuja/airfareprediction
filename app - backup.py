import shutil
from flask import Flask
from flask import request, redirect
from flask import render_template
import mysql.connector

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
			return render_template('rfg.html')
		else:
			print('out')
			return render_template('login.html')


@app.route("/prediction")
def prediction():
	return render_template('prediction.html')
	
@app.route("/start")
def start():
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
	return render_template('start.html', 
		tables=[shuffled_data_train.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse'), 
		shuffled_data_test.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse')], 
		fptr=Flight_Price_train, 
		fpte=Flight_Price_test,
		titles=[shuffled_data_train.columns.values,shuffled_data_test.columns.values])	

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


	# ##### The r2_score function computes R², the coefficient of determination. It provides a measure of how well future samples are likely to be predicted by the model. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a R^2 score of 0.0

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
	#print(dtVerticalScrollExample)
	return render_template('process.html',
		tables=[dtVerticalScrollExample.to_html(classes='data table  table-sm table-responsive table-bordered table-striped table-inverse')], titles=dtVerticalScrollExample.columns.values)

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