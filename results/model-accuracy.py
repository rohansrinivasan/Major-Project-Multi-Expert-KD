# Base Model Accuracy
y1_pred = model.evaluate([emg_test, acc_test, gyr_test], y_test, verbose=0)[1]
print("Base Model Accuracy:","{:.3f}%".format(y1_pred*100))

# Distilled Model Accuracy
y2_pred = distiller.evaluate([emg_test, acc_test, gyr_test], y_test, verbose=0)[1]
print("Distilled Model Accuracy","{:.3f}%".format(y2_pred*100))