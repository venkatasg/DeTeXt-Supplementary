import tensorflow as tf
import coremltools as ct
import pdb

if __name__=="__main__":

    with open('class_names.txt', 'r') as f:
        class_labels = f.read().splitlines()
        
    # Load the saved tf model first
    tf_model = tf.keras.models.load_model('saved_model3/')

    ct_model = ct.convert(tf_model, inputs=[ct.ImageType()], 
                        classifier_config = ct.ClassifierConfig(class_labels))
                        
    spec = ct_model.get_spec()
    # Rename the output dictionary to something sensible
    ct.utils.rename_feature(spec, 'Identity', 'classLabelProbs')
    ct.utils.rename_feature(spec, 'input_1', 'drawing')
    ctmodel = ct.models.MLModel(spec)

    # Set feature descriptions (these show up as comments in XCode)
    ctmodel.input_description["drawing"] = "Input drawing to be classified"
    ctmodel.output_description["classLabel"] = "Most likely symbol"
    ctmodel.output_description["classLabelProbs"] = "Probability scores for each symbol"

    # Set model author name
    ctmodel.author = "Venkata S Govindarajan"

    # Set the license of the model
    ctmodel.license = "MIT License"

    # Set a short description for the Xcode UI
    ctmodel.short_description = "Detects the most likely LaTeX mathematical symbol corresponding to a drawing."

    # Set a version for the model
    ctmodel.version = "1.0"

    # Save model
    ctmodel.save("deTeXtf3.mlmodel")
