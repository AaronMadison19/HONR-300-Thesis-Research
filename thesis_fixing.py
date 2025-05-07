import os
import time
import torch
import numpy as np
import cv2
import json
from PIL import Image
from glob import glob
import argparse
from tqdm import tqdm

start_time = time.time()

# Set random seed for reproducibility
torch.manual_seed(2023)
np.random.seed(2023)

# Define dataset class
class CustomDataset:
    def __init__(self, root, data, threshold=0.35):
        self.data = data
        self.threshold = threshold
        self.im_paths = [im_path for im_path in sorted(glob(f"{root}/{data}/*/*")) if "jpg" in im_path or "jpeg" in im_path or "png" in im_path]
        
        self.cls_names, self.cls_counts, count = {}, {}, 0
        for idx, im_path in enumerate(self.im_paths):
            class_name = self.get_class(im_path)
            if class_name not in self.cls_names:
                self.cls_names[class_name] = count
                self.cls_counts[class_name] = 1
                count += 1
            else:
                self.cls_counts[class_name] += 1
        
        # Print found classes
        print(f"Found {len(self.cls_names)} classes: {self.cls_names}")
        print(f"Found {len(self.im_paths)} images")
    
    def get_class(self, path):
        return os.path.dirname(path).split("/")[-1]
    
    def __len__(self):
        return len(self.im_paths)

    def __getitem__(self, idx):
        im_path = self.im_paths[idx]
        im = Image.open(im_path).convert('RGB')
        gt = self.cls_names[self.get_class(im_path)]
        
        # Apply transforms manually
        im = im.resize((224, 224))
        im_array = np.array(im) / 255.0
        # Normalize with ImageNet stats
        im_array = (im_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        # Convert to tensor manually
        im_tensor = torch.FloatTensor(im_array.transpose(2, 0, 1))
        
        # Apply augmentation
        if self.data == "train":
            rand = np.random.rand()
            if rand > self.threshold:
                # Rotate 90 degrees (transpose and flip)
                im_tensor = torch.rot90(im_tensor, 1, [1, 2])
            elif rand > self.threshold and rand < 2 * self.threshold:
                # Rotate -90 degrees
                im_tensor = torch.rot90(im_tensor, 3, [1, 2])
        
        return im_tensor, gt

# Simple CNN model
class SimpleCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        # A smaller model hopefully more suitable for Raspberry Pi
        self.features = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            torch.nn.Conv2d(16, 32, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            torch.nn.Conv2d(32, 64, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            torch.nn.Conv2d(64, 128, kernel_size=3, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            
            torch.nn.AdaptiveAvgPool2d((1, 1))
        )
        
        self.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Save class mapping without compression
def save_class_mapping(class_mapping, filepath):
    inverse_mapping = {v: k for k, v in class_mapping.items()}
    with open(filepath, 'w') as f:
        json.dump(inverse_mapping, f)

# Load class mapping without compression
def load_class_mapping(filepath):
    with open(filepath, 'r') as f:
        return {int(k): v for k, v in json.load(f).items()}

# Create datasets and dataloaders manually
def get_data_batches(dataset, batch_size, shuffle=False):
    indices = list(range(len(dataset)))
    if shuffle:
        np.random.shuffle(indices)
    
    for start_idx in range(0, len(dataset), batch_size):
        batch_indices = indices[start_idx:start_idx + batch_size]
        batch_samples = [dataset[i] for i in batch_indices]
        
        # Combine samples into a batch
        batch_data = [s[0] for s in batch_samples]
        batch_labels = [s[1] for s in batch_samples]
        
        yield torch.stack(batch_data), torch.tensor(batch_labels)

def train_model(root, output_dir, batch_size=4, epochs=10, learning_rate=1e-4):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Use CPU for training on Raspberry Pi
    device = "cpu"
    print(f"Using device: {device}")
    
    # Get datasets
    train_dataset = CustomDataset(root=root, data="train")
    test_dataset = CustomDataset(root=root, data="test")
    
    # Split train into train and validation
    train_size = int(0.9 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    
    # Create random indices for train and validation
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create subset datasets
    class SubsetDataset:
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
        
        def __len__(self):
            return len(self.indices)
        
        def __getitem__(self, idx):
            return self.dataset[self.indices[idx]]
    
    train_ds = SubsetDataset(train_dataset, train_indices)
    val_ds = SubsetDataset(train_dataset, val_indices)
    
    print(f"Training samples: {len(train_ds)}")
    print(f"Validation samples: {len(val_ds)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Save class mapping
    class_mapping = {v: k for k, v in train_dataset.cls_names.items()}
    save_class_mapping(class_mapping, os.path.join(output_dir, "class_mapping.json"))
    
    # Create model
    model = SimpleCNN(num_classes=len(train_dataset.cls_names))
    model = model.to(device)
    
    # Define loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    
    # Training and validation loop
    print("Starting training...")
    # Changed: Track best accuracy instead of best loss
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss, train_correct = 0.0, 0
        train_total = 0
        
        for batch_data, batch_labels in tqdm(get_data_batches(train_ds, batch_size, shuffle=True), 
                                           total=len(train_ds)//batch_size, 
                                           desc=f"Epoch {epoch+1}/{epochs} - Training"):
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            
            # Forward pass
            outputs = model(batch_data)
            loss = loss_fn(outputs, batch_labels)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Calculate accuracy
            _, predicted = torch.max(outputs.data, 1)
            train_total += batch_labels.size(0)
            train_correct += (predicted == batch_labels).sum().item()
            train_loss += loss.item() * batch_data.size(0)
        
        train_loss = train_loss / train_total
        train_acc = train_correct / train_total
        
        # Validation phase
        model.eval()
        val_loss, val_correct = 0.0, 0
        val_total = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in tqdm(get_data_batches(val_ds, batch_size), 
                                               total=len(val_ds)//batch_size, 
                                               desc=f"Epoch {epoch+1}/{epochs} - Validation"):
                batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
                
                # Forward pass
                outputs = model(batch_data)
                loss = loss_fn(outputs, batch_labels)
                
                # Calculate accuracy
                _, predicted = torch.max(outputs.data, 1)
                val_total += batch_labels.size(0)
                val_correct += (predicted == batch_labels).sum().item()
                val_loss += loss.item() * batch_data.size(0)
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
        
        print(f"Epoch {epoch+1}/{epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Changed: Save the model with the best validation accuracy
        if val_acc > best_accuracy:
            best_accuracy = val_acc
            torch.save(model.state_dict(), os.path.join(output_dir, "age_best_model.pth"))
            print(f"Model saved to {os.path.join(output_dir, 'age_best_model.pth')} with validation accuracy: {val_acc:.4f}")
    
    # Test the model
    print("Testing the model...")
    model.load_state_dict(torch.load(os.path.join(output_dir, "age_best_model.pth")))
    model.eval()
    
    test_correct, test_total = 0, 0
    with torch.no_grad():
        for batch_data, batch_labels in tqdm(get_data_batches(test_dataset, batch_size=1), 
                                           total=len(test_dataset), 
                                           desc="Testing"):
            batch_data, batch_labels = batch_data.to(device), batch_labels.to(device)
            outputs = model(batch_data)
            _, predicted = torch.max(outputs.data, 1)
            test_total += batch_labels.size(0)
            test_correct += (predicted == batch_labels).sum().item()
    
    test_acc = test_correct / test_total
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Make model ready for inference
    model.eval()
    return model

class AgeDetector:
    def __init__(self, model_path, class_mapping_path):
        # Load class mapping
        self.class_mapping = load_class_mapping(class_mapping_path)
        num_classes = len(self.class_mapping)
        
        # Create and load model
        self.model = SimpleCNN(num_classes=num_classes)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        
        # Face detection using OpenCV's Haar cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        print(f"Loading face cascade from: {cascade_path}")
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            print("Warning: Haar cascade file not found or invalid!")
            # Try an alternative location common on Raspberry Pi
            alt_path = "/usr/share/opencv4/haarcascades/haarcascade_frontalface_default.xml"
            if os.path.exists(alt_path):
                print(f"Trying alternative path: {alt_path}")
                self.face_cascade = cv2.CascadeClassifier(alt_path)
    
    def preprocess_image(self, image):
        # Convert to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # Convert to PIL
        pil_image = Image.fromarray(image_rgb)
        # Resize
        pil_image = pil_image.resize((224, 224))
        # Convert to numpy and normalize
        img_array = np.array(pil_image) / 255.0
        img_array = (img_array - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        # Convert to tensor
        img_tensor = torch.FloatTensor(img_array.transpose(2, 0, 1)).unsqueeze(0)
        return img_tensor
    
    def detect_faces(self, frame):
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5,
            minSize=(30, 30)
        )
        return faces
    
    def predict_age(self, face_img):
        # Preprocess face image
        face_tensor = self.preprocess_image(face_img)
        
        # Make prediction
        with torch.no_grad():
            outputs = self.model(face_tensor)
            _, predicted = torch.max(outputs, 1)
            predicted_idx = predicted.item()
            
            print(f"Class mapping: {self.class_mapping}")
            print(f"Type of class mapping: {type(self.class_mapping)}")
            print(f"Predicted index: {predicted_idx}")
            print(f"Type of predicted index: {type(predicted_idx)}")
            
            predicted_idx_str = str(predicted_idx)
            
            # Fix: Directly access the class mapping by index
            #if str(predicted_idx) in self.class_mapping:
            if predicted_idx_str in self.class_mapping:
                
                #age_class = self.class_mapping[str(predicted_idx)]
                age_class = self.class_mapping[predicted_idx_str]

            else:
                try:
                    age_class = self.class_mapping[predicted_idx]
                except (KeyError, TypeError):
                    
                    age_class = "Unknown"
                    print(f"Unknown class index: {predicted_idx}")
                    print(f"Available indices: {list(self.class_mapping.keys())}")
        
        return age_class
    
    def start_camera(self):
        print("Initializing camera...")
        
        # Initialize the camera
        # On Raspberry Pi, Pi camera module is probably needed // ill come back to that later :)
        try:
            #cap = cv2.VideoCapture(0)
            cap = cv2.VideoCapture("test.h264")
            if not cap.isOpened():
                print("Error: Could not open camera. Trying alternative...")
                # Try the Pi camera if available
                cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
        except Exception as e:
            print(f"Error initializing camera: {e}")
            return
        
        if not cap.isOpened():
            print("Error: Could not open any camera.")
            return
        
        print("Camera initialized. Press 'q' to quit.")
        
        try:
            while True:
                # Capture frame
                ret, frame = cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break
                
                # Make a copy for drawing
                output_frame = frame.copy()
                
                # Detect faces
                faces = self.detect_faces(frame)
                
                # Process detected faces
                for (x, y, w, h) in faces:
                    # Extract face
                    face_img = frame[y:y+h, x:x+w]
                    
                    # Skip if face is too small
                    if face_img.size == 0 or w < 30 or h < 30:
                        continue
                    
                    # Predict age
                    try:
                        age_class = self.predict_age(face_img)
                        
                        # Draw rectangle and age
                        cv2.rectangle(output_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(output_frame, f"Age: {age_class}", (x, y-10),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                    except Exception as e:
                        print(f"Error predicting age: {e}")
                
                # Display frame
                cv2.imshow('Age Detection', output_frame)
                
                # Break loop if 'q' is pressed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("Interrupted by user")
        except Exception as e:
            print(f"Error in camera loop: {e}")
        finally:
            # Release resources
            cap.release()
            cv2.destroyAllWindows()
            print("Camera released.")

def main():
    parser = argparse.ArgumentParser(description='Age Detection System for Raspberry Pi')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the age detection model')
    train_parser.add_argument('--data-dir', type=str, required=True, help='Path to the dataset directory')
    train_parser.add_argument('--output-dir', type=str, default='saved_models', help='Directory to save the model')
    train_parser.add_argument('--batch-size', type=int, default=4, help='Batch size for training')
    train_parser.add_argument('--epochs', type=int, default=10, help='Number of epochs for training')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    
    # Detect command
    detect_parser = subparsers.add_parser('detect', help='Run age detection using the camera')
    detect_parser.add_argument('--model-path', type=str, default='saved_models/age_best_model.pth', help='Path to the trained model')
    detect_parser.add_argument('--class-mapping', type=str, default='saved_models/class_mapping.json', help='Path to the class mapping file')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        print("Starting training process...")
        train_model(
            root=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
            epochs=args.epochs,
            learning_rate=args.lr
        )
        print("Training completed!")
        
    elif args.command == 'detect':
        print("Starting age detection with camera...")
        try:
            detector = AgeDetector(
                model_path=args.model_path,
                class_mapping_path=args.class_mapping
            )
            detector.start_camera()
        except Exception as e:
            print(f"Error in detection: {e}")
        
    else:
        parser.print_help()

if __name__ == "__main__":
    main()

end_time  = time.time()

execution_time = end_time - start_time
print(f"Execution time: {execution_time:.4f} seconds")
