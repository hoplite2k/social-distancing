#execute this script only once to download ssdlite_mobilenet_v2 
import tensorflow as tf
import pathlib

def download_model(model_name):
    model_url = 'http://download.tensorflow.org/models/object_detection/' + model_name + '.tar.gz'

    model_dir = tf.keras.utils.get_file(
        fname=model_name, 
        origin=model_url,
        untar=True,
        cache_dir=pathlib.Path('.tmp').absolute()
    )

    #model = tf.saved_model.load(model_dir + '/saved_model')
    #return model

MODEL_NAME = 'ssdlite_mobilenet_v2_coco_2018_05_09'
download_model(MODEL_NAME)

