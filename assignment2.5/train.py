import argparse
from ultralytics import YOLO

def main(args):
    # Use the name of the YOLO model that you have chosen, e.g., 'yolov8n.pt'    
    model_name = args.model_name
    # Specify the path to your custom dataset
    dataset_path = args.dataset_path
    # Adjust the number of epochs and other hyperparameters according to your needs
    epochs = args.epochs
    #imgsz = args.img_size

    # Load a model
    model = YOLO(model_name)  # Load a pretrained YOLO model

    # Train the model
    model.train(
        data=dataset_path,
        epochs=epochs,
        # imgsz=imgsz,
    )
    
    '''
    In YOLOv8, after the training, the model is automatically saved to the "runs/train/" directory with incrementing run directories.
    For example, the first run will be saved under "runs/train/exp", the second run under "runs/train/exp2", and so on.
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLO model')
    parser.add_argument('--model_name', type=str, default='yolov8x.pt', help='Path to the pretrained YOLO model')
    parser.add_argument('--dataset_path', type=str, default='datasets/yolo/my_database.yaml', help='Path to the YAML file')
    parser.add_argument('--epochs', type=int, default=4, help='Number of training epochs')

    args = parser.parse_args()
    main(args)
