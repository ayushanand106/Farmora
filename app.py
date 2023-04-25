from flask import Flask, render_template, request ,jsonify
import numpy as np
import pickle

app = Flask(__name__)
model = pickle.load(open("model", "rb"))
crop=None

@app.route("/")
def home():
    return  render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    output = round(prediction[0], 2)

    answer=int(output)
    if(answer==0):
        crop=="Jowar"
    elif(answer==1):
        crop=="Bajra"
    elif(answer==2):
        crop=="Maize"
    elif(answer==3):
        crop=="Cotton"
    elif(answer==4):
        crop=="Groundnut"
    elif(answer==5):
        crop=="Jute"
    elif(answer==6):
        crop=="Sugarcane"
    elif(answer==7):
        crop=="Turmeric"
    elif(answer==8):
        crop=="Urad Dal"
    elif(answer==9):
        crop=="Coffee"
    elif(answer==10):
        crop=="Rice"
    elif(answer==11):
        crop=="Gram"
    elif(answer==12):
        crop=="Pea"
    elif(answer==13):
        crop=="Barley"
    elif(answer==14):
        crop=="Potato"
    elif(answer==15):
        crop=="Tomato"
    elif(answer==16):
        crop=="Onion"
    elif(answer==17):
        crop=="Sunflower"
    elif(answer==18):
        crop=="Mustard"
    elif(answer==19):
        crop=="=Wheat"


    return render_template("index.html", prediction_text="Best crop to be grown in the field is ${}".format(answer))

if __name__ == "__main__":
    app.run(debug=True)
