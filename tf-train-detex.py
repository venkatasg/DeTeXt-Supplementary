import tensorflow as tf

EPOCHS = 1
BATCH_SIZE = 128
LEARNING_RATE = 0.003
SEED=1220

if __name__ == '__main__':	
	# Load train and validation data
	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
								'/Volumes/detext/drawings/',
							    color_mode="grayscale",
  								seed=SEED,
  								batch_size=BATCH_SIZE,
								labels='inferred',
								label_mode='int',
								image_size=(200,300))
	
	# Get the class names
	class_names = train_ds.class_names
	num_classes = len(class_names)

	# Create model
	model = tf.keras.applications.MobileNetV3Small(
    	input_shape=(200,300,1), alpha=1.0, minimalistic=False, 
    	include_top=True, weights=None, input_tensor=None, classes=num_classes,
    	pooling=None, classifier_activation="softmax",
    	include_preprocessing=True)

	# Compile model
	model.compile(
		   optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
           loss=tf.keras.losses.SparseCategoricalCrossentropy(),
           metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])

    # Training
	model.fit(x=train_ds, epochs=EPOCHS)
	
	# Testing
	hist = model.evaluate(x=train_ds)
	print(hist)
	model.save('./saved_model3/')
