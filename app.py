import pickle
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
app = Flask(__name__)

df = pd.read_csv('cardata.csv')

model = pickle.load(open("model.pkl","rb"))

@app.route('/')
def home():
    year = sorted(df['Year'].unique())
    fuel_type = df['Fuel_Type'].unique()
    seller_type = df['Seller_Type'].unique()
    transmission_type = df['Transmission'].unique()
    owner_type = df['Owner'].unique()
    return render_template('index.html',year=year,fuel_type=fuel_type,seller_type=seller_type,transmission_type=transmission_type,owner_type=owner_type)

@app.route('/predict',methods=['POST'])
def predict():
    year = int(request.form.get('year'))
    pre_price = float(request.form.get('pre_price'))
    km_driven = int(request.form.get('km_driven'))
    fuel_type = request.form.get('fuel_type')
    if fuel_type=="Petrol":
        fuel = 0
    elif fuel_type=="Diesel":
        fuel=1
    else:
        fuel=2
    seller_type = request.form.get('seller_type')
    if seller_type=="Dealer":
        seller = 0
    else:
        seller=1
    transmission_type = request.form.get('transmission_type')
    if transmission_type=="Manual":
        transmission = 0 
    else:
        transmission = 1
    owner_type = int(request.form.get('owner_type'))
    final = np.array([year,pre_price,km_driven,fuel,seller,transmission,owner_type])
    prediction = model.predict(final.reshape(1,-1))
    output = round(prediction[0],2)
    return render_template('index.html',prediction_text = f'Price would be around rs.{output} Lacs')


if __name__ == "__main__":
    app.run(debug=True)