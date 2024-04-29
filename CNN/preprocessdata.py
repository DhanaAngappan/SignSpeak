import os
import random
import shutil

def split_dataset(input_folder, output_folder, test_ratio=0.2):
    
    os.makedirs(output_folder, exist_ok=True)
    train_folder = os.path.join(output_folder, 'train')
    test_folder = os.path.join(output_folder, 'test')
    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    classes = os.listdir(input_folder)
    for class_name in classes:
        class_folder = os.path.join(input_folder, class_name)
        if os.path.isdir(class_folder):
           
            os.makedirs(os.path.join(train_folder, class_name), exist_ok=True)
            os.makedirs(os.path.join(test_folder, class_name), exist_ok=True)
            
            images = os.listdir(class_folder)
            
            random.shuffle(images)
            
            num_test = int(len(images) * test_ratio)

            
            for img in images[:num_test]:
                shutil.copy(os.path.join(class_folder, img), os.path.join(test_folder, class_name))

           
            for img in images[num_test:]:
                shutil.copy(os.path.join(class_folder, img), os.path.join(train_folder, class_name))

if __name__ == "__main__":
    input_folder = "MP_Data"
    output_folder = "preprocessed_dataset"
    split_dataset(input_folder, output_folder, test_ratio=0.2)
