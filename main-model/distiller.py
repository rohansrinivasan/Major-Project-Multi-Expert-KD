#Distill Student to Teacher

distiller = Distiller1(student=stu_model, teacher=model) #invokes the KD class
distiller.compile(
    optimizer=adam,
    metrics=['accuracy'],
    student_loss_fn=keras.losses.CategoricalCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=5,
    )
hist2= distiller.fit([emg_train, acc_train, gyr_train], y_train, epochs=50, batch_size=batch_size, verbose=True, validation_data=([emg_test, acc_test, gyr_test], y_test)) #, epochs=epochs,callbacks=[es]
hist_arr2 = np.array([hist2.history['accuracy'],hist2.history['student_loss'],hist2.history['distillation_loss'],hist2.history['val_accuracy'],hist2.history['val_student_loss']])

# Test Accuracy
plt.figure(5)
plt.plot(hist_arr2[0])
plt.ylabel('Test Accuracy (%)')
plt.xlabel('Epoch')
plt.grid(True)

# Validation Accuracy
plt.figure(2)
plt.plot(hist_arr2[3])
plt.ylabel('Train Accuracy (%)')
plt.xlabel('Epoch')
plt.grid(True)

# Test Loss
plt.figure(3)
plt.plot(hist_arr2[1])
plt.ylabel('Test Loss')
plt.xlabel('Epoch')
plt.grid(True)

# Validation Loss
plt.figure(4)
plt.plot(hist_arr2[4])
plt.ylabel('Train Loss')
plt.xlabel('Epoch')
plt.grid(True)

# Distillation Loss
plt.figure(4)
plt.plot(hist_arr2[2])
plt.ylabel('Teacher over Student Distillation Loss')
plt.xlabel('Epoch')
plt.grid(True)