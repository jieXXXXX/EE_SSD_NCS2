# EE_SSD_NCS2
1.	VGG16 backbone file is named “VGG16_sdd_3_exit”, Mobilenet-v2 backbone file is named “mobilenet_ssd_3 exit”.
2.	Prepare for the datasets
download the VOC2007 datasets file, it should include 5 file:
Annotations, ImageSets, JPEGImages, SegmentationClass, SegmentationObject.
Use “create_data_lists.py” to generate the .json file
the output should contain: TRAIN_images.json, TRAIN_objects.json, TEST_images.json. TEST_objects.json, label_map.json
3.	Train the model file by using “train.py” and using pretrain model file ” checkpoint_ssdbest . pth ” to generate the training model file “output/file_name_training_time/ joint_weigh t.pth”
4.	Use “seperate_model.py” with “model_3_sub.py” to separate the joint_weight.pth into 5 parts. “backbone0.pth”, “backbone1.pth”, “backbone2.pth”,”sub1.pth”,”sub2.pth”.
5.	Use “transfer.py” to transfer the “.pth” file into “.onnx” file
6.	After setting the environment of OpenVINO, go to the path” openvino_2021.4.752\depl oyment_tools\model_optimizer\mo.py” and copy the “.onnx” file into this path , and execute the “python mo.py --input_model file_name.onnx” command in cmd window. 
7.	After this step , it will generate the “.xml”, ”.bin”, ”.map” file 
8.	copy the “.xml” file and “.bin ” file into the “different model version/weight” file and store the detected image into “test_detect_image” file .
9.	Use “detect_NCS2_3_EXIT.py” to detect image or use “detect_evaluate.py” to evaluate the model


