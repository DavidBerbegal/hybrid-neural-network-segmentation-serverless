import tensorflow as tf

modelo = tf.keras.models.load_model("C:\\Users\\admlocal\\Desktop\\Suma\\Azure-Segmentado\\funcion-verificacion-modelo-local\\modelo_sin_entrenar.h5")
#modelo = tf.keras.models.load_model("C:\\Users\\admlocal\\Desktop\\Suma\\Azure-Segmentado\\funcion-verificacion-modelo-local\\modelo_entrenado.h5")

modelo.build(input_shape=(None, 2))

modelo.summary()
