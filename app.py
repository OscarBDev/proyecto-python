from flask import Flask, request, jsonify, render_template, redirect, url_for, session
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2
#capa presonalizada
import tensorflow_hub as hub 
from keras.utils import get_custom_objects
#IMPORTAMOS LA CLAVE SECRETA
from config import SECRET_KEY
#base de datos
from flask_sqlalchemy import SQLAlchemy
#para la tabla de categorias donde mostraremos un producto al azar
from sqlalchemy.sql.expression import func


#inicializamos flask
app = Flask(__name__)

#establecemos una clave secerta para las sesiones 
app.secret_key = SECRET_KEY

#configuramos la conexion a mysql 
app.config['SQLALCHEMY_DATABASE_URI'] = 'mysql+pymysql://root:@localhost/ollas'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

#iniciamos la base de datos
db = SQLAlchemy(app)

#definimos el modelo de nuestras tablaa
#aqui usamos producto y no olla por que es confuso
class Productos(db.Model):
    __tablename__ = 'ollas'
    ID_OLLAS = db.Column(db.Integer, primary_key=True)
    IMAGEN = db.Column(db.LargeBinary)  # Cambié a LargeBinary para almacenar imágenes
    NOMBRE = db.Column(db.String(30), nullable=False)
    COMENSALES = db.Column(db.Integer)
    CAPACIDAD = db.Column(db.String(5))
    COLOR = db.Column(db.String(15))
    MEDIDA = db.Column(db.String(8))
    STOCK = db.Column(db.Integer)
    PRECIO_UNITARIO = db.Column(db.Numeric(10, 2))
    ID_CATEGORIA = db.Column(db.Integer, db.ForeignKey('CATEGORIA.ID_CATEGORIA'))
    categoria = db.relationship('Categoria', backref='ollas')  # Relación con CATEGORIA

class Categoria(db.Model):
    __tablename__ = 'CATEGORIA'
    ID_CATEGORIA = db.Column(db.Integer, primary_key=True)
    NOMBRE = db.Column(db.String(30), nullable=False)
    

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
#etiquetas con las etiquteas de las categorias
etiquetas_clase_inv = {
    "Asador": 1,  
    "Cacerola": 2,
    "Olla presión": 3,
    "Sartén": 4,
    "Wok": 5
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

#mandamos todos los datos de productos a productos.html
@app.route('/productos.html')
def productos():
    #consultamos todos los productos de la base de datos 
    productos = Productos.query.all()
    
    #enviamos estos datos a productos.html
    return render_template('productos.html', productos = productos)

#mandamos los datos de la pbase de datos para la vista o el modelo de categorias.html
@app.route('/categorias.html')
def productos_por_categoria():
    #obtenemos la todas las categorias
    categorias = Categoria.query.all()
    
    #creamos un diccionario para almacenar un producto por categoria
    productos_por_categoria = {}
    
    #Recorremos las categorias y seleccionamos un producto al azar 
    for categoria in categorias:
        producto_aleatorio = Productos.query.filter_by(ID_CATEGORIA=categoria.ID_CATEGORIA).order_by(func.rand()).first()
        productos_por_categoria[categoria.NOMBRE] = producto_aleatorio
        
    #mostramos el categorias.html
    return render_template('categorias.html', productos_por_categoria=productos_por_categoria)

#por aqui se enviara la prediccion de la imagen subida 
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file"
    
    file = request.files['file']
    
    if file:
        
        #Guardar y proprocesar la imagen
        img_path = './static/uploaded_images/' + file.filename
        file.save(img_path)
        img = preprocess_image(img_path)
        
        #hacemos la prediccion
        prediccion = modelo.predict(img)
        clase_index = np.argmax(prediccion[0])
        etiqueta = etiquetas_clase[clase_index]
        
        #return jsonify({'prediccion': etiqueta})
        
        #guardamos la prediccion en la sesion
        session['prediccion'] = etiqueta
        session['ruta_imagen'] = img_path  # Guarda la ruta de la imagen
        
        #redirigimos a la pagina de el resultado
        return redirect(url_for('resultado'))
    
#creamos una ruta para las predicciones el resultado basicamenete
@app.route('/resultado', methods=['GET'])
def resultado():
    prediccion = session.get('prediccion', 'No hay prediccion disponible.')
    ruta_imagen = session.get('ruta_imagen') # ruta de la imagen cargada
    
    # Obtener el ID_CATEGORIA basado en la etiqueta de clase predicha
    id_categoria = etiquetas_clase_inv.get(prediccion)

    # Consultar todos los productos que pertenecen a esa categoría
    productos = Productos.query.filter_by(ID_CATEGORIA=id_categoria).all()
     
    return render_template('resultado.html', prediccion=prediccion, productos=productos, ruta_imagen=ruta_imagen)
    
#ejecutamos
if __name__ == '__main__':
    app.run(debug=True)