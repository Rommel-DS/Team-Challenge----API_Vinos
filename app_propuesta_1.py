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
#HELLO DE RODRIGO, HAY QUE ADAPTARLO A NUESTRO TRABAJO (COMENTADO PARA PROBAR LANDPAGE CHATGPT)

@app.route("/", methods=["GET"])
def hello(): # Ligado al endopoint "/" o sea el home, con el método GET
    return """
    <h1>Bienvenido a la API del modelo que estima el alcohol de tu vino</h1>
    <h2>Diseñado por el grupo Vinos formado por: Rommel, Rodrigo, Guillermo y Jose Luis</h2>
    <h3>¿Qué hace esta API?</h3>
    <p>Esta API permite predecir el nivel de alcohol en un vino a partir de sus características químicas y organolépticas.</p>
    <p>Opciones disponibles:</p>
    <ul>
        <li><strong>/</strong> - Página inicial.</li>
        <a href="http://127.0.0.1:5000/" target="_blank">Página inicial</a>

        <li><strong>/api/v1/formulario_predict</strong> - Endpoint para introducir tu formulario de predicción de alcohol, hay un ejemplo sobre el que pueder sobrescribir los datos de tu vino y te calculará el alcohol que según nuestro modelo tiene tu vino. <br> Usa parámetros [fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol(si lo desconoces indicalo con un 0), quality, class_].</li>
        <a href="http://127.0.0.1:5000/api/v1/formulario_predict" target="_blank">Formulario prediccion alcohol</a>
               
        <li><strong>/api/v1/retrain</strong> - Endpoint para reentrenar el modelo con datos nuevos. <br> Busca automáticamente el archivo 'wines_retrain.csv' en la carpeta 'data'.</li>
        <a href="http://127.0.0.1:5000/api/v1/retrain/" target="_blank">Reentreno del modelo de prediccion alcohol</a>   
    </ul>
    <p>Para más información, accede a cada endpoint según corresponda.</p>
    """

#LANDPAGE CHATGPT PREDICCIÓN EN LANDPAGE
@app.route("/api/v1/formulario_predict", methods=["GET"])
def formulario_predict(): # Ligado al endopoint "/api/v1/formulario_predict", con el método GET
    return """
    <h1>Bienvenido a la API del modelo 'Alcohol en tu vino' 🍷</h1>
    <p>Introduce los valores del vino que quieras predecir su nivel de alcohol(te indicamos un ejemplo a sobre escribir, si desconoces el nivel de alcohol dejalo marcado como 0):</p>
    
    <form id="predictionForm">
        <label>Fixed Acidity: <input name="fixed_acidity" value=6.6></label><br>
        <label>Volatile Acidity: <input name="volatile_acidity" value=0.16></label><br>
        <label>Citric Acid: <input name="citric_acid" value=0.3></label><br>
        <label>Residual Sugar: <input name="residual_sugar" value=1.6></label><br>
        <label>Chlorides: <input name="chlorides" value=0.034></label><br>
        <label>Free Sulfur Dioxide: <input name="free_sulfur_dioxide" value=15.0></label><br>
        <label>Total Sulfur Dioxide: <input name="total_sulfur_dioxide" value=78.0></label><br>
        <label>Density: <input name="density" value=0.992></label><br>
        <label>pH: <input name="pH" value=3.38></label><br>
        <label>Sulphates: <input name="sulphates" value=0.44></label><br>
        <label>Alcohol: <input name="alcohol" value="0"></label><br>
        <label>Quality (0-10): <input name="quality" value=6></label><br>
        <label>Class (white/red): <input name="class_" value="white"></label><br><br>
        <button type="submit">Predecir</button>
    </form>

    <h3>Según nuestro modelo el alcohol de tu vino es de : <span id="predictionResult">---</span> Grados</h3>

    <script>
    document.getElementById("predictionForm").addEventListener("submit", function(event) {
        event.preventDefault();
        const formData = new FormData(event.target);
        const params = new URLSearchParams();
        for (const [key, value] of formData.entries()) {
            params.append(key, value);
        }
        fetch('/api/v1/predict?' + params.toString())
            .then(response => response.json())
            .then(data => {
                document.getElementById("predictionResult").innerText = data.prediction;
            })
            .catch(error => {
                document.getElementById("predictionResult").innerText = "Error: " + error;
            });
    });
    </script>
    """

# Enruta la funcion al endpoint /api/v1/predict
@app.route('/api/v1/predict', methods=["GET"])
def predict(): # Ligado al endpoint '/api/v1/predict', con el método GET
    with open('modelo_pipeline_reg.pkl', 'rb') as f:
        model = pickle.load(f)

    fixed_acidity = request.args.get('fixed_acidity', None)
    volatile_acidity = request.args.get('volatile_acidity', None)
    citric_acid = request.args.get('citric_acid', None)
    residual_sugar = request.args.get('residual_sugar', None)
    chlorides = request.args.get('chlorides', None)
    free_sulfur_dioxide = request.args.get('free_sulfur_dioxide', None)
    total_sulfur_dioxide = request.args.get("total_sulfur_dioxide", None)
    density = request.args.get('density', None)
    pH = request.args.get('pH', None)
    sulphates = request.args.get('sulphates', None)
    alcohol = request.args.get('alcohol', None)
    quality = request.args.get('quality', None)
    class_ = request.args.get('class_', None)


    print(fixed_acidity, volatile_acidity, citric_acid, residual_sugar, chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density, pH, sulphates, alcohol, quality, class_)
    print(type(fixed_acidity))

    if (fixed_acidity is None or
        volatile_acidity is None or
        citric_acid is None or
        residual_sugar is None or
        chlorides is None or
        free_sulfur_dioxide is None or
        total_sulfur_dioxide is None or
        density is None or
        pH is None or
        sulphates is None or
        alcohol is None or
        quality is None or
        class_ is None):
        return "Args empty, not enough data to predict"
    
    else:
    
        input_data = pd.DataFrame([{
            'fixed_acidity': float(fixed_acidity),
            'volatile_acidity': float(volatile_acidity),
            'citric_acid': float(citric_acid),
            'residual_sugar': float(residual_sugar),
            'chlorides': float(chlorides),
            'free_sulfur_dioxide': float(free_sulfur_dioxide),
            'total_sulfur_dioxide': float(total_sulfur_dioxide),
            'density': float(density),
            'pH': float(pH),
            'sulphates': float(sulphates),
            'alcohol': float(alcohol),
            'quality': int(quality),
            'class_': class_
        }])

        prediction = model.predict(input_data)
        
    
    return jsonify({'prediction': float(prediction[0])}) 


# # Enruta la funcion al endpoint /api/v1/retrain
# @app.route("/api/v1/retrain/", methods=["GET"])
# def retrain(): # Ligado al endpoint '/api/v1/retrain/', metodo GET
#     if os.path.exists("data/Advertising_new.csv"):
#         data = pd.read_csv('data/Advertising_new.csv')

#         X_train, X_test, y_train, y_test = train_test_split(data.drop(columns=['sales']),
#                                                         data['sales'],
#                                                         test_size = 0.20,
#                                                         random_state=42)

#         model = Lasso(alpha=6000)
#         model.fit(X_train, y_train)
#         rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
#         mape = mean_absolute_percentage_error(y_test, model.predict(X_test))
#         model.fit(data.drop(columns=['sales']), data['sales'])
#         with open('ad_model.pkl', 'wb') as f:
#             pickle.dump(model, f)
            
#         return f"Model retrained. New evaluation metric RMSE: {str(rmse)}, MAPE: {str(round(mape*100,2))}%"
#     else:
#         return f"<h2>New data for retrain NOT FOUND. Nothing done!</h2>"


if __name__ == '__main__':
    app.run(debug=True)
