import tensorflow.compat.v2 as tf

tf.enable_v2_behavior()

from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()

from tensorflow.python.compiler.mlcompute import mlcompute
mlcompute.set_mlc_device(device_name='any')

EPOCHS = 5
BATCH_SIZE = 64
LEARNING_RATE = 0.003
SIGDIG=3
SEED=1220
THRESHOLD=0.001

normalization_layer = tf.keras.layers.experimental.preprocessing.Rescaling(1./255)

if __name__ == '__main__':
	
	#Set random seeds
	
	# Load the Data
	train_ds = tf.keras.preprocessing.image_dataset_from_directory(
								'../detexify-data/drawings/', 
								validation_split=0.2,
							    subset="training",
  								seed=SEED,
								labels='inferred')

	# Get the class names
    # class_ids = {v:k for k,v in data.class_to_idx.items()}
	class_names = train_ds.class_names
	num_classes = len(class_names)

	# Transform to values between 0 and 1
	train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
	
	val_ds = tf.keras.preprocessing.image_dataset_from_directory(
								'../detexify-data/drawings/', 
								validation_split=0.2,
							    subset="validation",
  								seed=SEED,
								labels='inferred')
	val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
    
	# Create model
# 	model = tf.keras.applications.MobileNetV2(
# 		input_shape=(256,256,3), alpha=1.0, include_top=True, weights=None,
# 		input_tensor=None, pooling=None, classes=num_classes, 
# 		classifier_activation='softmax')
	
	model = tf.keras.applications.MobileNetV3Small(
    	input_shape=(256,256,3), alpha=1.0, minimalistic=False, include_top=True,
    	weights=None, input_tensor=None, classes=1000, pooling=None,
    	dropout_rate=0, classifier_activation='softmax')
	
	# Compile model
	model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
	model.summary()	
              
    # Training
	history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS)