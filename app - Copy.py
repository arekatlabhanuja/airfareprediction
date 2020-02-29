from flask import Flask
from flask import request
from flask import render_template
import mysql.connector

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
			return render_template('prediction.html')
		else:
			return render_template('login.html')


@app.route("/prediction")
def prediction():
	return render_template('prediction.html')
 
if __name__ == "__main__":
	app.run(host='0.0.0.0', port=5000)