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



Usage

Prepare the data:


# reassemble and unzip the data
cd datasets
zip -F pix2code_datasets.zip --out datasets.zip
unzip datasets.zip

cd ../model

# split training set and evaluation set while ensuring no training example in the evaluation set
# usage: build_datasets.py <input path> <distribution (default: 6)>
./build_datasets.py ../datasets/ios/all_data
./build_datasets.py ../datasets/android/all_data
./build_datasets.py ../datasets/web/all_data

# transform images (normalized pixel values and resized pictures) in training dataset to numpy arrays (smaller files if you need to upload the set to train your model in the cloud)
# usage: convert_imgs_to_arrays.py <input path> <output path>
./convert_imgs_to_arrays.py ../datasets/ios/training_set ../datasets/ios/training_features
./convert_imgs_to_arrays.py ../datasets/android/training_set ../datasets/android/training_features
./convert_imgs_to_arrays.py ../datasets/web/training_set ../datasets/web/training_features





Train the model:



mkdir bin
cd model

# provide input path to training data and output path to save trained model and metadata
# usage: train.py <input path> <output path> <is memory intensive (default: 0)> <pretrained weights (optional)>
./train.py ../datasets/web/training_set ../bin

# train on images pre-processed as arrays
./train.py ../datasets/web/training_features ../bin



evaluate the model:

cd model
python  eval.py ../bin pix2code  ../datasets/web/eval_set





Generate code for a single GUI image:

mkdir code
cd model

# generate DSL code (.gui file), the default search method is greedy
# usage: sample.py <trained weights path> <trained model name> <input image> <output path> <search method (default: greedy)>
./sample.py ../bin pix2code ../test_gui.png ../code

# equivalent to command above
./sample.py ../bin pix2code ../test_gui.png ../code greedy

# generate DSL code with beam search and a beam width of size 3
./sample.py ../bin pix2code ../test_gui.png ../code 3



Compile generated code to target language:

cd compiler

# compile .gui file to Android XML UI
./android-compiler.py <input file path>.gui

# compile .gui file to iOS Storyboard
./ios-compiler.py <input file path>.gui

# compile .gui file to HTML/CSS (Bootstrap style)
./web-compiler.py <input file path>.gui




