# blstm-pix2code
Use blstm to modify the pix2code model，using the Bidirectional LSTM improved model,the accuracy rate can reach 83%.


the Original paper model github address https://github.com/tonybeltramelli/pix2code

This model modifies the code of the pix2code core model. Please refer to the above link for the database.


requirements

Keras==2.1.2
numpy==1.13.3
opencv-python==3.3.0.10
h5py==2.7.1
tensorflow==1.4.0



#Usage

Prepare the data:


cd datasets
zip -F pix2code_datasets.zip --out datasets.zip
unzip datasets.zip

cd ../model

./build_datasets.py ../datasets/ios/all_data
./build_datasets.py ../datasets/android/all_data
./build_datasets.py ../datasets/web/all_data


./convert_imgs_to_arrays.py ../datasets/ios/training_set ../datasets/ios/training_features
./convert_imgs_to_arrays.py ../datasets/android/training_set ../datasets/android/training_features
./convert_imgs_to_arrays.py ../datasets/web/training_set ../datasets/web/training_features





#Train the model:



mkdir bin
cd model


./train.py ../datasets/web/training_set ../bin


./train.py ../datasets/web/training_features ../bin



#evaluate the model:

cd model
python  eval.py ../bin pix2code  ../datasets/web/eval_set





Generate code for a single GUI image:

mkdir code
cd model

./sample.py ../bin pix2code ../test_gui.png ../code


./sample.py ../bin pix2code ../test_gui.png ../code greedy


./sample.py ../bin pix2code ../test_gui.png ../code 3



#Compile generated code to target language:

cd compiler


./android-compiler.py <input file path>.gui


./ios-compiler.py <input file path>.gui


./web-compiler.py <input file path>.gui




