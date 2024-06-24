from flask import Flask,request,render_template
import pickle
import numpy as np  

app = Flask(__name__)

def load_model():
    with open('customerSat.pkl','rb') as file:
        data = pickle.load(file)
    return data

files = load_model()

model = files['model']
encoder_cust = files['encoder_cust']
encoder_type = files['encoder_type']
encoder_class = files['encoder_class']
scaler = files['scaler']

@app.route('/')
def homepage():
    return render_template('satisfaction.html')

@app.route('/predict',methods=['POST'])
def do_prediction():
    customer_type = request.form.get('customer type')
    age = request.form.get('age')
    type_of_travel = request.form.get('type of travel')
    class_ = request.form.get('class')
    flight_distance = request.form.get('flight distance')
    seat_comfort = request.form.get('seat comfort')
    convenient = request.form.get('departure/arrival time convenient')
    food = request.form.get('food and drink')
    gate_location = request.form.get('gate location')
    wifi = request.form.get('inflight wifi service')
    entertainment = request.form.get('inflight entertainment')
    online_support = request.form.get('online support')
    online_booking = request.form.get('ease of online booking')
    on_board = request.form.get('on-board service')
    leg_room = request.form.get('leg room service')
    baggage = request.form.get('baggage handling')
    checkin = request.form.get('checkin service')
    cleanliness = request.form.get('cleanliness')
    online_boarding = request.form.get('online boarding')
    delaydep = request.form.get('departure delay in minute')
    delayarr = request.form.get('arrival delay in minute')
    
    x = np.array([[customer_type,age,type_of_travel,class_,flight_distance,seat_comfort,convenient,food,gate_location,wifi,entertainment,online_support,online_booking,on_board,leg_room,baggage,checkin,cleanliness,online_boarding,delaydep,delayarr]])
    x[:,0] = encoder_cust.transform(x[:,0])
    x[:,2] = encoder_type.transform(x[:,2])
    x[:,3] = encoder_class.transform(x[:,3])
    
    x = scaler.transform(x)
    prediction = model.predict(x)
    
    msg = 'Customer is likely to be satisfied' if prediction == 1 else 'Customer is likely to be dissatisfied'
    
    return render_template('satisfaction.html',text = msg)

if __name__ == '__main__':
    app.run(host="0.0.0.0")