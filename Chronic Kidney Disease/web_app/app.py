from flask import Flask,render_template,url_for,request
from flask_bootstrap import Bootstrap 
from sklearn.externals import joblib
from sklearn.preprocessing import RobustScaler
import pandas as pd

app = Flask(__name__)
Bootstrap(app)

model = open("C:/Users/gauta/Desktop/Guvi/CKD/web_app/1model.pkl")
clf = joblib.load(model)

def label_encode_rbc(x):
    switcher = {
        "normal": 1,
        "abnormal":2
    }
    return switcher.get(x, 0)
def label_encode_pc(x):
    switcher = {
        "normal": 1,
        "abnormal":2
    }
    return switcher.get(x, 0)
def label_encode_pcc(x):
    switcher = {
        "present": 1,
        "notpresent":2
    }
    return switcher.get(x, 0)
def label_encode_ba(x):
    switcher = {
        "present": 1,
        "notpresent":2
    }
    return switcher.get(x, 0)
def label_encode_htn(x):
    switcher = {
        "no": 1,
        "yes":2
    }
    return switcher.get(x, 0)
def label_encode_dm(x):
    switcher = {
        "no": 1,
        "yes":2
    }
    return switcher.get(x, 0)
def label_encode_cad(x):
    switcher = {
        "no": 1,
        "yes":2
    }
    return switcher.get(x, 0)
def label_encode_appet(x):
    switcher = {
        "no": 1,
        "poor":2,
        "good":3
    }
    return switcher.get(x, 0)
def label_encode_pe(x):
    switcher = {
        "no": 1,
        "yes":2
    }
    return switcher.get(x, 0)
def label_encode_ane(x):
    switcher = {
        "no": 1,
        "yes":2
    }
    return switcher.get(x, 0)
    
def encoding(data):
    for i in data["rbc"]:
        data["rbc"] = int(label_encode_rbc(str(i)))
    for i in data["pc"]:
        data["pc"] = int(label_encode_pc(str(i)))
    for i in data["pcc"]:
        data["pcc"] = int(label_encode_pcc(str(i)))
    for i in data["ba"]:
        data["ba"] = int(label_encode_ba(str(i)))
    for i in data["htn"]:
        data["htn"] = int(label_encode_htn(str(i)))
    for i in data["dm"]:
        data["dm"] = int(label_encode_dm(str(i)))
    for i in data["cad"]:
        data["cad"] = int(label_encode_cad(str(i)))
    for i in data["appet"]:
        data["appet"] = int(label_encode_appet(str(i)))
    for i in data["pe"]:
        data["pe"] = int(label_encode_pe(str(i)))
    for i in data["ane"]:
        data["ane"] = int(label_encode_ane(str(i)))
    return data


@app.route('/')
def index():
	return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = int(request.form['age'])
        bp = float(request.form['bp'])
        sg = float(request.form['sg'])
        al = float(request.form['al'])
        su = float(request.form['su'])
        rbc = request.form['rbc']
        pc = request.form['pc']
        pcc = request.form['pcc']
        ba = request.form['ba']
        bgr = float(request.form['bgr'])
        bu = float(request.form['bu'])
        sc = float(request.form['sc'])
        sod = float(request.form['sod'])
        pot = float(request.form['pot'])
        hemo = float(request.form['hemo'])
        pcv = float(request.form['pcv'])
        wbcc = float(request.form['wbcc'])
        rbcc = float(request.form['rbcc'])
        htn = request.form['htn']
        dm = request.form['dm']
        cad = request.form['cad']
        appet = request.form['appet']
        pe = request.form['pe']
        ane = request.form['ane']
        
        data = {'age' : age,'bp' : bp,'sg' : sg,'al' : al,'su' : su,'rbc' : rbc,'pc' : pc,
                'pcc' : pcc,'ba' : ba,'bgr' : bgr,'bu' : bu,'sc' : sc,'sod' : sod,'pot' : pot,
                'hemo' : hemo,'pcv' : pcv,'wbcc' : wbcc,'rbcc' : rbcc,'htn' : htn,'dm' : dm,
                'cad' : cad,'appet' : appet,'pe' : pe,'ane' : ane}
        index = [0]
        data = pd.DataFrame(data, index=index)
        data = encoding(data)
        data_robust = pd.DataFrame(RobustScaler().fit_transform(data), columns=data.columns)
        
        y_pred = clf.predict(data_robust)
        
        return render_template('results.html', name = y_pred)
    

if __name__ == '__main__':
        app.run(debug=True)