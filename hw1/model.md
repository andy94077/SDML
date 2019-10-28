# pretrain
## model12-24single.weight
	batch_size = 3

## model18-24_layer_by_layer.weight
	batch_size = 4
	val_loss: 0.9809 - val_MLM_loss: 0.2519 - val_NSP_loss: 4.7918

# model

## model24
	model12-24single.weight
	```
	x = model.layers[-9].output
	x = Lambda(lambda model: model[:, 0])(x)
	x = Dense(1024)(x)
	x = BatchNormalization()(x)
	x = Activation('tanh')(x)
	x = Dense(1024)(x)
	x = BatchNormalization()(x)
	x = Activation('tanh')(x)
	```
	val_loss: 0.3439 - val_f1_acc: 0.6563 - val_acc: 0.6360

## model24-2
	model12-24single.weight
	```
	x = Lambda(lambda model: model[:, 0])(x)
	x = Dense(1024)(x)
	x = BatchNormalization()(x)
	x = Activation('tanh')(x)
	x = Dense(1024)(x)
	x = BatchNormalization()(x)
	x = Activation('tanh')(x)
	```
	val_loss: 0.3420 - val_f1_acc: 0.6582 - val_acc: 0.6390

## model24-3
	model18-24_layer_by_layer.weight

## model 24_1e-4
	lr=1e-4

## modelv3
	```
	x = model.layers[-9].output
	x = Lambda(lambda model: model[:, 0])(x)
	x = Dense(1024)(x)
	x = BatchNormalization()(x)
	x = Activation('tanh')(x)
	x = Dense(1024)(x)
	x = BatchNormalization()(x)
	x = Activation('tanh')(x)
	```
	val_loss: 0.3398 - val_f1_acc: 0.6620 - val_acc: 0.6540

## model_cite
	```
	x = model.layers[-9].output
	x = Lambda(lambda model: model[:, 0])(x)
	c = Input(shape=(4,))
	x = Concatenate()([x, c])
	x = Dense(1024)(x)
	x = BatchNormalization()(x)
	x = Activation('tanh')(x)
	x = Dense(1024)(x)
	x = BatchNormalization()(x)
	x = Activation('tanh')(x)
	Output_layer = Dense(3, activation = 'sigmoid')(x)
	```
	val_loss: 0.3347 - val_f1_acc: 0.6660 - val_acc: 0.6570

## model_cite_one_layer
	lr=1e-4
	```
	x = model.layers[-9].output
	x = Lambda(lambda model: model[:, 0])(x)
	x = BatchNormalization()(x)
	c = Input(shape=(4,))
	x = Concatenate()([x, c])
	Output_layer = Dense(3, activation = 'sigmoid')(x)
	```
	val_loss: 0.3293 - val_f1_acc: 0.6705 - val_acc: 0.6270


