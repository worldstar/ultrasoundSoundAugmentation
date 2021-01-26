from core.Model.LeNet_Functional_Model import buildLeNetModel
from core.CustomDataGenerator import CustomDataGenerator

if __name__ == "__main__":

    log_dir="./model/"
    
    inputs=(150,150,3)
    batch_size=32
    epochs=10
    num_classes = 2

    datagen=CustomDataGenerator(fun="CLAHE_Color",clahenum=40,dtype=int)

    train_generator = datagen.flow_from_directory(
        './DCdata/train',
        target_size=(150, 150),
        batch_size=32,
        class_mode='categorical')
    val_generator = datagen.flow_from_directory(
            './DCdata/val',
            target_size=(150, 150),
            batch_size=32,
            class_mode='categorical')

    model = buildLeNetModel(inputs, num_classes)

    # checkpoint = ModelCheckpoint(log_dir + "ep{epoch:03d}-loss{loss:.3f}-val_loss{val_loss:.3f}.h5",
    #     monitor='val_loss', save_weights_only=True, save_best_only=True, period=1)

    # callbacks_list = [checkpoint]

    model.fit_generator(
        train_generator,
        steps_per_epoch=10,
        epochs=epochs,
        validation_data=val_generator,
        validation_steps=10
        # , callbacks=[callbacks_list]
        )

