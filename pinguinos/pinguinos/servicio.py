import pickle

from flask import Flask, request, jsonify

pinguino_tipo = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}

def predict_single(penguin,dv,model):
    penguin_std = dv.transform(penguin)
    y_pred = model.predict(penguin_std)[0]
    y_prob = model.predict_proba(penguin_std)[0][y_pred]
    return(y_pred,y_prob)


def predict(dv, model):
    pinguino = request.get_json()
    especie_index, probabilidad = predict_single(pinguino,dv,model)
    
    especie = pinguino_tipo.get(especie_index)
    
    result = {
        'pinguino': especie,
        'probabilidad': float(probabilidad)
    }
    return jsonify(result)

app = Flask('pinguino')

@app.route('/predict_lr', methods=['POST'])
def predict_lr():
    with open('modelos/lr.pck', 'rb') as f:
        dv, model = pickle.load(f)
        return predict(dv, model)
@app.route('/predict_svm', methods=['POST'])
def predict_svm():
    with open('modelos/svm.pck', 'rb') as f:
        dv, model = pickle.load(f)
        return predict(dv, model)
@app.route('/predict_dt', methods=['POST'])
def predict_dt():
    with open('modelos/tree_model.pck', 'rb') as f:
        dv, model = pickle.load(f)
        return predict(dv, model)
@app.route('/predict_knn', methods=['POST'])
def predict_knn():
    with open('modelos/knn.pck', 'rb') as f:
        dv, model = pickle.load(f)
        return predict(dv, model)
    
if __name__ == '__main__':
    app.run(debug=True,port=8000)