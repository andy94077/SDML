## model24
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


