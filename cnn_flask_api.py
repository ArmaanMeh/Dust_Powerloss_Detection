import flask
import numpy as np
import cv2
import tensorflow as tf 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

class CompatDepthwiseConv2D(tf.keras.layers.DepthwiseConv2D): 
    def __init__(self, *args, groups=1, **kwargs): 
        if groups != 1:
            raise ValueError(f"Expected groups=1 in saved model, got {groups}")
        super().__init__(*args, **kwargs)


model = load_model(
    "Models/Mobilenet.h5",
    compile=False,
    custom_objects={"DepthwiseConv2D": CompatDepthwiseConv2D}, 
)
app = flask.Flask(__name__) 
 
@app.route("/")
def index(): 
    # Render the home page template with the image upload form
    return flask.render_template("home.html") 

@app.route("/predict", methods=["POST"])
def predict():
    file = flask.request.files.get("file")
    if file is None or not file.filename: 
        flask.abort(400, description="Choose an image file to classify.")
 
    img = cv2.imdecode(np.frombuffer(file.read(), dtype=np.uint8), cv2.IMREAD_COLOR) 
    if img is None: 
        flask.abort(400, description="The uploaded file is not a readable image.")

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
 
    x = image.img_to_array(img) 
    x = np.expand_dims(x, axis=0) 
    x = x / 255 

    predictions = model.predict(x, verbose=0)
    dust_score = float(predictions[0][0]) 
    label = "Dusty" if dust_score > 0.5 else "Clean"

    return flask.render_template( 
        "result.html", 
        label=label,
        dust_score=dust_score,
        filename=file.filename,
    ) 

if __name__ == "__main__":
    app.run(debug=True)