import tensorflow as tf
import coremltools as ct

if __name__=="__main__":
    mlmodel = ct.convert('saved_model/', inputs=[ct.ImageType()])