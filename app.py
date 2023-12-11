from flask import Flask,render_template,request,redirect,session, send_file
from flask_pymongo import PyMongo
import datetime
from pymongo import MongoClient
import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
import scipy.optimize as spop
import matplotlib.pyplot as plt
import seaborn
from datetime import datetime
import yfinance as yf
from pykalman import KalmanFilter
from statsmodels.tsa.stattools import coint
from io import BytesIO
import base64
import seaborn as sns
from tkinter import * 
from tkinter import ttk 
from kiteconnect import KiteConnect
import sqlite3
import threading
global kite 
from pprint import pprint
from stock import StockPairsTrading
global li
global st_name
global st_sl
global st_qty

app=Flask(__name__)

# app.config["SECRET_KEY"]="profitforge"
# app.config["MONGO_URI"] = "mongodb+srv://bondjames181920:Shreyash18@cluster0.d3op3bw.mongodb.net/?retryWrites=true&w=majority"
# app.config["MONGO_DBNAME"]="db"

client = MongoClient("mongodb+srv://bondjames181920:Shreyash18@cluster0.d3op3bw.mongodb.net/?retryWrites=true&w=majority")
db = client['profitforge'] 
# mongo = PyMongo(app)



@app.route('/')
def home():
	if "number" in session:

		return render_template('index.html', logged_in=True)
	else:
		return render_template('index.html', logged_in=False)

   

@app.route('/logge',methods=['GET', 'POST'])
def logge():
	number=request.form.get('number')
	password=request.form.get('pass')

	apikey=request.form.get('apikey')
	connectZerodha2(apikey)
	# user = auth.sign_in_with_email_and_password(email,password)
	# username = db.child(user['localId']).child("Username").get().val()

	# if user:
	# 	return render_template('dashboard.html')    #return f"Welcome {username}!"
	user=db['users']
	note = user.find_one({"number":number , "password":password})
	if note:
		session['number']=number
		return redirect("/")

	else:
		return "Invalid credentials"
	


@app.route('/regg',methods=['GET', 'POST'])
def regg():
	email=request.form.get('email')
	password=request.form.get('pass')
	number=request.form.get('number')
	
	# user = auth.create_user_with_email_and_password(email, password)
	# user = auth.sign_in_with_email_and_password(email, password)
	# db.child(user['localId']).child("Username").set(username)
	user=db['users']
	user.insert_one({"email":email,"password":password,"number":number})
	session['number']=number


	return redirect("/")


@app.route('/asset_data',methods=['GET','POST'])
def asset_data():
	name=request.form.get('name_asset')
	number_share=request.form.get('no_share')
	comments=request.form.get('Comments')
	date=request.form.get('date')
	user=db['assets']
	if "number" in session:
		user.insert_one({"name":name,"number":number_share,"comments":comments,"id":session['number'],"date":date})
	# 	return name
	# if name is None:
	# 	return "No input"
	# else:
	# 	return name
	else:
		return redirect("/gotodash")
	return redirect("/gotodash")


@app.route('/login')
def login():
	# if 'prn' in session:
	# 	return "you are logged in as" + session['prn']
    return render_template("login.html")

@app.route('/gotodash')
def gotodash():
	if 'number' in session:
		return render_template("dashboard.html")
	else:
		redirect("/login")

@app.route('/getname',methods=['GET','POST'])
def getname():
	if 'number' in session:
		number=session['number']
		user=db['users']
		asset=db['assets']
		name=user.find_one({"number":number})
		if name:
			user_name=name.get('email')
			numbers=name.get('number')
		if 'number' in session:
			notes=list(asset.find({"id":session['number']}))

		return render_template("dashboard.html",user_name=user_name,numbers=numbers,notes=notes,logged_in=True)

	else:
		return "login to see dashboard"
	
	# user=db['users']
	# if 'prn' in session:
	# 	notes=list(user.find({"number":session['number']}))
	# 	print(notes)
	# 	return notes
	# else:
	# 	redirect("/login")

@app.route('/findpair2',methods=['POST','GET'])
def findpair2():
	a1=request.form.get('a1')
	a2=request.form.get('a2')
	try:
		# pair , df=find_pairs([a1,a2])
		# # stock1, stock2 = pair
		# plt.figure(figsize=(10, 5))
		# plt.plot(df.index, df[pair[0][0]], label=pair[0][0])
		# plt.plot(df.index, df[pair[0][1]], label=pair[0][1])
		# plt.xlabel('Date')
		# plt.ylabel('Value')
		# plt.title('Asset Performance Over Time')
		# plt.legend()

		# # Save the plot to a BytesIO object
		# img_buf = BytesIO()
		# plt.savefig(img_buf, format='png')
		# img_buf.seek(0)
		# img_data = base64.b64encode(img_buf.read()).decode()
		image_path = 'path/to/your/image.jpg'

    # Read the image file and encode it as base64
		with open("outputs\performance.png", "rb") as image_file:
			encoded_image = base64.b64encode(image_file.read()).decode('utf-8')
		spt = StockPairsTrading(
    	start="2015-11-22"
    # end="2023-11-22",
		)
		r1 = spt.backtest((a1, a2))
		r2 = spt.latest_signal((a1, a2))
		r2_list=list(r2.items())
		pprint(r1)
		pprint(r2)



						

		

		# img_data2=signals(pair,df)
		#calculate spread

		return render_template("findpair2.html", r2=r2,encoded_image=encoded_image)
	except:
		return "No pair found"

	# if pair is None:
	# 	return render_template("findpair2.html", pairs=pair, ans=False, graph_data=img_data)
	

	# # plt.figure(figsize=(15, 7))
	# # for pair in pair:
	# # 	stock1, stock2 = pair
	# # 	plt.plot(data.index, data[stock1] / data[stock1][0], label=stock1)
	# # 	plt.plot(data.index, data[stock2] / data[stock2][0], label=stock2)
	# # plt.xlabel("Date")
	# # plt.ylabel("Normalized Adj Close")
	# # plt.legend(loc="upper left")
	# # plt.title("Normalized Adj Close Prices of Cointegrated Pairs")
	# # plt.savefig("{}/pairs_value_plot.png".format(outputs_dir_path))
	# # plt.show()
	# return render_template("findpair2.html", pairs=pair, ans=True,graph_data=img_data)

	# return find_pairs([a1,a2])

@app.route("/logout",methods=['GET','POST'])	
def logout():
	if "number" in session:
		session.pop("number",None)
		return redirect("/")

@app.route("/trade_data",methods=['POST','GET'])
def trade_data():
	a1=request.form.get('a2')
	#add custom date
	input_data1=request.form.get('d1')
	input_data2=request.form.get('d2')
	# date_object1 = datetime.strptime(input_data1, "%m-%d-%Y")
	# output_date1 = date_object1.strftime("%Y-%m-%d")
	# date_object2 = datetime.strptime(input_data2, "%m-%d-%Y")
	# output_date2 = date_object2.strftime("%Y-%m-%d")
	start="2023-09-01"
	end="2023-12-01" 	
	ticker = yf.Ticker(a1)
	data = ticker.history(period="1d", start=input_data1, end=input_data2)
	df = data.reset_index()
	data_dict_list = df.to_dict(orient='records')
	#print(df)
	if data is None:
		return render_template("trade.html")
	else:
		return render_template("trade.html", dataa=data_dict_list)


@app.route("/getlivedata",methods=['POST'])
def getlivedata():
	global li
	# req_token=request.form.get('apikey')
	
	try:
		# s=connectZerodha(req_token)
		# index=request.form.get('index')
		name=request.form.get('name')
		
		li=[]
		li.append("NSE:"+name)
		li.append("NSE:TATASTEEL")
		instruments = ["NSE:TATASTEEL", "NSE:RELIANCE", "NSE:INFY", "NSE:HDFC"]
		data=get_live_data2(instruments=li)
		# return data
		return render_template('livedata.html', live_data=data)
		
		# return li

	except:
		return "enter valid"
		

@app.route("/buysell2",methods=['POST'])
def buysell2():
	global kite
	global st_name
	global st_sl
	global st_qty
	# req_token=request.form.get('apikey')
	st_name=request.form.get('name')
	cap=request.form.get('cap')
	loss=request.form.get('loss')
	
	ordertype=request.form.getlist('ordertype')
	trans=request.form.get('trans')
	# s=connectZerodha(req_token)

	try:
		# s=connectZerodha(req_token)
		# session['name']=s
		# while 'name' in session:
		margin, st_qty, st_sl, lp=generateFields(loss,st_name,cap)
		# if request.form['action']=='Buy':
		# 	placeOrder(name,qty,sl,lp)
		# 	return "successful"
		
		return render_template('buysell.html',margin=margin,qty=st_qty, sl=st_sl,lp=lp)


	except:
		return "no such value"


@app.route("/buy_sell",methods=['POST'])
def buy_sell():
	global kite
	global st_name
	global st_sl
	global st_qty
	ordertype=request.form.get("ordertype")
	trans=request.form.get("trans")
	try:
		place_Order(st_name, st_qty, st_sl)
		gif_path = 'outputs\out2.gif'
		return send_file(gif_path, mimetype='image/gif')
		# return "yes"
	except Exception as e:
		print(e)
		# gif_path = 'outputs\out2.gif'
		# return send_file(gif_path, mimetype='image/gif')
		return "no"
	




# def connect():
# 	req_token=request.form.get('apikey')
# 	connectZerodha(req_token)
		



@app.route('/about')
def about():
    return render_template("about.html")


@app.route('/l2')
def l2():
    return render_template("l2.html")

@app.route('/r2')
def r2():
    return render_template("r2.html")

@app.route('/hhhh')
def hhhh():
    return render_template("findpair2.html")

@app.route('/buysell') 
def buysell():
	return render_template("buysell.html")

@app.route('/trade')
def trade():
    return render_template("trade.html")

@app.route('/livedata')
def livedata():
    return render_template("livedata.html")

@app.route('/gopair')
def gopair():
    return render_template("findpair.html")



@app.route('/addasset')
def addasset():
    return render_template("addasset.html")

@app.route('/gottobuy2')
def gottobuy2():
    return render_template("buy2.html")

@app.route('/service')
def service():
    return render_template("service.html")

@app.route('/gotoreg')
def gotoreg():
    return render_template("regis.html")

start="2015-11-22"
end="2023-12-01"
outputs_dir_path="static\images"
column: str = "Adj Close"

def find_pairs( tickers: list) :
        columns = []
        for i in tickers:
            columns.append((column, i))
        df = (
            yf.download(tickers, start=start, end=end)[columns]
            .set_axis(tickers, axis="columns")
            .fillna(method="ffill")
            .dropna()
        )
        _, pvalues, pairs =_find_cointegrated_pairs(df)
        plt.figure(figsize=(15, 7))
        seaborn.heatmap(
            pvalues,
            xticklabels=tickers,
            yticklabels=tickers,
            cmap="RdYlGn_r",
            mask=(pvalues >= 0.05),
        )
        plt.savefig("{}/pairs.png".format(outputs_dir_path))
        plt.clf()
        plt.close()

        # plot_pairs(df, pairs)
        return pairs,df 

def _find_cointegrated_pairs( data: pd.DataFrame):
        n = data.shape[1]
        score_matrix = np.zeros((n, n))
        pvalue_matrix = np.ones((n, n))
        keys = data.keys()
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                S1 = data[keys[i]]
                S2 = data[keys[j]]
                result = coint(S1, S2)
                score = result[0]
                pvalue = result[1]
                score_matrix[i, j] = score
                pvalue_matrix[i, j] = pvalue
                if pvalue < 0.05:
                    pairs.append((keys[i], keys[j]))
        return score_matrix, pvalue_matrix, pairs

# def generate_trading_signals(pair, df):
#     ratio = df[pair[0]] / df[pair[1]]
#     zscore = (ratio - ratio.mean()) / ratio.std()

#     # Generate buy/sell signals
#     signals = pd.Series(index=df.index, dtype=int)
#     signals[zscore < -1] = 1  # Buy signal
#     signals[zscore > 1] = -1  # Sell signal

#     # Plot the signals and save to BytesIO
#     plt.figure(figsize=(10, 5))
#     sns.lineplot(x=df.index, y=df[pair[0]], label=pair[0])
#     sns.lineplot(x=df.index, y=df[pair[1]], label=pair[1])
#     plt.scatter(signals[signals == 1].index, df[pair[0]][signals == 1], marker="^", color="g", label="Buy Signal")
#     plt.scatter(signals[signals == -1].index, df[pair[0]][signals == -1], marker="v", color="r", label="Sell Signal")
#     plt.xlabel('Date')
#     plt.ylabel('Value')
#     plt.title(f'Trading Signals for {pair[0]} and {pair[1]}')
#     plt.legend()

#     # Save the plot to a BytesIO object
#     img_buf = BytesIO()
#     plt.savefig(img_buf, format='png')
#     img_buf.seek(0)
#     img_data = base64.b64encode(img_buf.read()).decode()

#     # Close the plot
#     plt.clf()
#     plt.close()

#     return img_data


def signals(pair,df):
		ratio = df[pair[0]] / df[pair[1]]
		zscore = (ratio - ratio.mean()) / ratio.std()

		# Generate buy/sell signals
		signals = pd.Series(index=df.index, dtype=int)
		signals[zscore < -1] = 1  # Buy signal
		signals[zscore > 1] = -1  # Sell signal

        # Plot the signals on a graph
		plt.figure(figsize=(15, 7))	
		plt.plot(df[pair[0]], label=pair[0])
		plt.plot(df[pair[1]], label=pair[1])
		plt.scatter(signals[signals == 1].index, df[pair[0]][signals == 1], marker="^", color="g", label="Buy Signal")
		plt.scatter(signals[signals == -1].index, df[pair[0]][signals == -1], marker="v", color="r", label="Sell Signal")
		plt.title(f"Pair Trading Signals for {pair[0]} and {pair[1]}")
		plt.xlabel("Date")
		plt.ylabel("Price")
		plt.legend()
		img_buf = BytesIO()
		plt.savefig(img_buf, format='png')
		img_buf.seek(0)
		img_data = base64.b64encode(img_buf.read()).decode()
		return img_data



def connectZerodha(req_token):
    global kite
    # logging.basicConfig(level=logging.DEBUG)
    kite = KiteConnect(api_key="mp89u8jdghogm6nz")
    data = kite.generate_session(req_token, api_secret="mvg47t5pbcgeok5df0ge4bru6tcmvw1a")
    kite.set_access_token(data["access_token"])
    profile = kite.profile()
    return profile["user_name"]


stop = False


def connectZerodha2(req_token):
	global kite
	api_key = "mp89u8jdghogm6nz"
	api_secret = "mvg47t5pbcgeok5df0ge4bru6tcmvw1a"

    # Check if the user is already logged in


    # If not logged in, generate a new session
	kite = KiteConnect(api_key="mp89u8jdghogm6nz")
	data = kite.generate_session(req_token, api_secret="mvg47t5pbcgeok5df0ge4bru6tcmvw1a")
	kite.set_access_token(data["access_token"])
    # Store the access token in the session
	session['access_token'] = data["access_token"]
	session.permanent = True  # Set the session to be permanent (you may adjust this based on your needs)

    # Fetch and return user information
	profile = kite.profile()
	return profile["user_name"]

def get_live_data2(instruments):
	global kite
	data = []
	quote = kite.quote(instruments)
	for instrument_key in instruments:
		if instrument_key in quote:
			if "ohlc" in quote[instrument_key] and "close" in quote[instrument_key]["ohlc"]:
				last_price = quote[instrument_key]["last_price"]
				change = ((last_price - quote[instrument_key]["ohlc"]["close"]) / quote[instrument_key]["ohlc"]["close"]) * 100
				data.append({
                    "instrument": instrument_key,
                    "high": quote[instrument_key]["ohlc"]["high"],
                    "low": quote[instrument_key]["ohlc"]["low"],
                    "volume": quote[instrument_key]["volume"],
                    "last_price": last_price,
                    "change": "{:.2f}".format(change),
                    "change_color": "green" if change > 0 else "red"
                })	

	return data

def zscore(series):
    return (series - series.mean()) / np.std(series)


def get_live_data(instruments):
    data = []
    quote = kite.quote(instruments)

    for instrument_key in instruments:
        if instrument_key in quote:
            if "ohlc" in quote[instrument_key] and "close" in quote[instrument_key]["ohlc"]:
                last_price = quote[instrument_key]["last_price"]
                change = ((last_price - quote[instrument_key]["ohlc"]["close"]) / quote[instrument_key]["ohlc"]["close"]) * 100
                data.append({
                    "instrument": instrument_key,
                    "high": quote[instrument_key]["ohlc"]["high"],
                    "low": quote[instrument_key]["ohlc"]["low"],
                    "volume": quote[instrument_key]["volume"],
                    "last_price": last_price,
                    "change": "{:.2f}".format(change),
                    "change_color": "green" if change > 0 else "red"
                })
            else:
                data.append({
                    "instrument": instrument_key,
                    "high": "N/A",
                    "low": "N/A",
                    "volume": "N/A",
                    "last_price": "N/A",
                    "change": "N/A",
                    "change_color": "black"
                })
        else:
            data.append({
                "instrument": instrument_key,
                "high": "N/A",
                "low": "N/A",
                "volume": "N/A",
                "last_price": "N/A",
                "change": "N/A",
                "change_color": "black"
            })

    return data

def generateFields(loss,name,capital):
	global kite
	order_param_single = [{
        "exchange": "NSE",
        "tradingsymbol":name,
        "transaction_type": "BUY",
        "variety": "CO",
        "product": "MIS",
        "order_type": "MARKET",
        "quantity": 1
        }]
	margin_detail = kite.order_margins(order_param_single)
	margin_tot=margin_detail[0]["total"]
	qty=(int)((float)(capital)/margin_detail[0]["total"])

	sl=(float)(loss)/qty
	instruments="NSE:"+name
	quotes=kite.quote(instruments)
	# if(trans=='Buy'):
	# 	sl=quotes["NSE:"+name.get()]["last_price"]-sl
	# else:
	# 	sl=quotes["NSE:"+name.get()]["last_price"]+sl 
	# 	sl=(float)("{:.2f}".format(sl))
	sl=((int)(sl*100)-((int)(sl*100))%5)/100
	ltp=quotes["NSE:"+name]["last_price"]
	return margin_tot,qty, sl, ltp


def place_Order(stockSymbol, qty, sl):
    global kite
    quotes=kite.quote("NSE:"+stockSymbol)
    ltp=quotes["NSE:"+stockSymbol]["last_price"]
    orderType=kite.ORDER_TYPE_MARKET
    transType=kite.TRANSACTION_TYPE_BUY
    # if(radio.get()==2):
    #     orderType=kite.ORDER_TYPE_LIMIT
    # if(radio2.get()==2):
    #     transType=kite.TRANSACTION_TYPE_SELL
    limitPrice=(float)(ltp)
    kite.place_order(
        variety=kite.VARIETY_CO,
        exchange=kite.EXCHANGE_NSE,
        tradingsymbol=stockSymbol,
        transaction_type=transType,
        quantity=qty,
        product=kite.PRODUCT_CNC,
        order_type=orderType,
        validity=kite.VALIDITY_DAY,
        trigger_price=21,
        price=limitPrice
    )



if __name__ == '__main__':
	app.secret_key='profitforge'
	app.run(debug=True)
	
