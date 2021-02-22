from utills.converter import convert_from_save_model

train_name = "t2021_02_20_11_37_mb2_224"
root_path = "save/"+train_name
model_path = "save/"+train_name+"/checkpoint"
tflite_path = "save/"+train_name+"/model.tflite"

convert_from_save_model(model_path,tflite_save_path=tflite_path)
