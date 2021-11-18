import tensorflow as tf
import pdb

EPOCHS = 1
BATCH_SIZE = 64
LEARNING_RATE = 0.003
SIGDIG=3
SEED=1220
THRESHOLD=0.001

if __name__ == '__main__':
	
	#Set random seeds
	
	# Load the Data
	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
								'/Volumes/detext/drawings/',
							    color_mode="grayscale",
  								seed=SEED,
  								batch_size=BATCH_SIZE,
								labels='inferred',
								label_mode='int',
								image_size=(200,300))

	# Get the class names
    # class_ids = {v:k for k,v in data.class_to_idx.items()}
	class_names = train_ds.class_names
	num_classes = len(class_names)

	# Create model - MNetV3 automatically does rescaling and normalization automatically
	model = tf.keras.applications.MobileNetV3Small(
    	input_shape=(200,300,1), alpha=1.0, minimalistic=False, 
    	include_top=True, weights=None, input_tensor=None, classes=num_classes,
    	pooling=None, dropout_rate=0, classifier_activation=None)
	
	# Compile model
	model.compile(optimizer='adam',
           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
           metrics=['accuracy'])
# 	model.summary()	

    # Training
	model.fit(train_ds, epochs=EPOCHS)
	
	model.save('./saved_model3/')