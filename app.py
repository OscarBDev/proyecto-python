from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
#capa presonalizada
import tensorflow_hub as hub 
from keras.utils import get_custom_objects
#IMPOTAMOS LA CLAVE SECRETA
from config import SECRET_KEY
#base de datos
from flask_sqlalchemy import SQLAlchemy

#inicializamos flask
app = Flask(__name__)

#establecemos una clave secerta para las sesiones 
app.secret_key = SECRET_KEY

#configuramos la conexion a mysql 
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/ollas'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#iniciamos la base de datos
db = SQLAlchemy(app)

#definimos el modelo de nuestra tabla
class Productos(db.Model):
    __tablename__ = 'ollas'
    ID_OLLAS = db.Column(db.Integer, primary_key=True)
    NOMBRE = db.Column(db.String(30), nullable=False)
    MEDIDA = db.Column(db.String(8), nullable=False)
    IMAGEN = db.Column(db.String(255), nullable=False)
    

class CustomMobileNetV2(tf.keras.layers.Layer):
    def __init__(self, trainable=True, **kwargs):
        super(CustomMobileNetV2, self).__init__(trainable=trainable, **kwargs)
        self.mobilenet = hub.KerasLayer('https://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/4',input_shape=(224, 224, 3))
        self.mobilenet.trainable = False  # Congelar la capa

    def call(self, inputs):
        return self.mobilenet(inputs)

# Asegúrate de registrar la clase personalizada
get_custom_objects().update({'CustomMobileNetV2': CustomMobileNetV2})

#Cargamos el modelo 
modelo = tf.keras.models.load_model('modelo_categorias.h5')

#etiquetas de las categorias 
etiquetas_clase = {
    0: "Asador",
    1: "Cacerola",
    2: "Olla presión",
    3: "Sartén",
    4: "Wok"
}

#procesamos la imagen para el modelo
def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img = np.array(img).astype(float) / 255
    img = cv2.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    return img

#cargamos la pagina atraves de una ruta llevamos datos de la base de datos tambien
@app.route('/')
def home():
    ollas_lista = Productos.query.all()
    return render_template('index.html', ollas = ollas_lista)

#por aqui se enviara la prediccion de la imagen subida 
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file"
    
    file = request.files['file']
    
    if file:
        
        #Guardar y proprocesar la imagen
        img_path = './uploaded_images/' + file.filename
        file.save(img_path)
        img = preprocess_image(img_path)
        
        #hacemos la prediccion
        prediccion = modelo.predict(img)
        clase_index = np.argmax(prediccion[0])
        etiqueta = etiquetas_clase[clase_index]
        
        #return jsonify({'prediccion': etiqueta})
        
        #guardamos la prediccion en la sesion
        session['prediccion'] = etiqueta
        
        #redirigimos a la pagina de el resultado
        return redirect(url_for('resultado'))
    
#creamos una ruta para las predicciones el resultado basicamenete
@app.route('/resultado', methods=['GET'])
def resultado():
    prediccion = session.get('prediccion', 'No hay prediccion disponible.')
    return render_template('resultado.html', prediccion=prediccion)
    
#ejecutamos
if __name__ == '__main__':
    app.run(debug=True)