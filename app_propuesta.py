from flask import Flask, jsonify, request
import os
import pickle
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
import numpy as np
import seaborn as sns

# os.chdir(os.path.dirname(__file__))

app = Flask(__name__)


# Enruta la landing page (endpoint /)
@app.route("/", methods=["GET"])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    return """
    <h1>Bienvenido a la API del modelo alcohol en tu vino, diseñado por el grupo Vinos formado por: Rommel, Rodrigo, Guillermo y Jose Luis</h1>
    <p>Opciones disponibles:</p>
    <ul>
        <li><strong>/</strong> - Página inicial.</li>
        <li><strong>/api/v1/predict</strong> - Endpoint para realizar predicciones. <br> Usa parámetros 'tv', 'radio' y 'newspaper' en la URL para predecir.</li>
        <li><strong>/api/v1/retrain</strong> - Endpoint para reentrenar el modelo con datos nuevos. <br> Busca automáticamente el archivo 'Advertising_new.csv' en la carpeta 'data'.</li>
    </ul>
    <p>Para más información, accede a cada endpoint según corresponda.</p>
    """

# Enruta la funcion al endpoint /api/v1/predict
@app.route('/api/v1/predict', methods=["GET"])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET
    with open('modelo_pipeline_reg.pkl', 'rb') as f:
        model = pickle.load(f)

    tv = request.args.get('tv', None)
    radio = request.args.get('radio', None)
    newspaper = request.args.get('newspaper', None)

    print(tv,radio,newspaper)
    print(type(tv))

    if tv is None or radio is None or newspaper is None:
        return "Args empty, not enough data to predict"
    else:
        prediction = model.predict([[float(tv),float(radio),float(newspaper)]])
    
    return jsonify({'predictions': prediction[0]})


# Enruta la funcion al endpoint /api/v1/retrain
@app.route("/api/v1/retrain/", methods=["GET"])
def retrain(): # Ligado al endpoint '/api/v1/retrain/', metodo GET
    if os.path.exists("data/Advertising_new.csv"):
        data = pd.read_csv('data/Advertising_new.csv')

        X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
                                                        data['sales'],
                                                        test_size = 0.20,
                                                        random_state=42)

        model = Lasso(alpha=6000)
        model.fit(X_train, y_train)
        rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
        mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
        model.fit(data.drop(columns=['sales']), data['sales'])
        with open('ad_model.pkl', 'wb') as f:
            pickle.dump(model, f)
            
        return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(round(mape*100,2))}%"
    else:
        return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"


if __name__ == '__main__':
    app.run(debug=True)
