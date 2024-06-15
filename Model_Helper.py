import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms, models
from time import time

def create_model(architecture, hidden_units, learning_rate):
    print("\nInside create_model")
    
    if architecture =='vgg16':
        model = models.vgg16(pretrained = True)
        input_units = 25088
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.classifier = nn.Sequential(
              nn.Linear(input_units, hidden_units),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(hidden_units, 256),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(256, 102),
              nn.LogSoftmax(dim = 1)
            )
        
        optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
        
        
    elif architecture =='vgg13':
        model = models.vgg13(pretrained = True)
        input_units = 25088
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.classifier = nn.Sequential(
              nn.Linear(input_units, hidden_units),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(hidden_units, 256),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(256, 102),
              nn.LogSoftmax(dim = 1)
            )
        
        optimizer = optim.Adam(model.classifier.parameters(), lr = learning_rate)
        
        
    elif architecture =='resnet50':
        model = models.resnet50(pretrained = True)
        input_units = 2048
        
        for param in model.parameters():
            param.requires_grad = False
        
        model.fc = nn.Sequential(
              nn.Linear(input_units, hidden_units),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(hidden_units, 256),
              nn.ReLU(),
              nn.Dropout(p=0.2),
              nn.Linear(256, 102),
              nn.LogSoftmax(dim = 1)
            )
        
        optimizer = optim.Adam(model.fc.parameters(), lr = learning_rate)
    else:
        raise ValueError('Architecture not supported ' + architecture)
        
    return model, optimizer

def train_model(model, optimizer, epochs, train_loader, valid_loader, gpu):
    print("\nInside train_model")
    
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    print("\nUsing device: {}".format(device))

    criterion = nn.NLLLoss()
    
    model.to(device)
    
    step_counter = 0
    print_every = 10
    training_loss = 0

    for epoch in range(epochs):
        for inputs, labels in train_loader:
            step_counter += 1
            inputs, labels = inputs.to(device), labels.to(device)
            
            #Training 
            optimizer.zero_grad()
            
            logps = model.forward(inputs)
            current_loss = criterion(logps, labels)
            current_loss.backward()
            optimizer.step()

            training_loss += current_loss.item()
         
            if step_counter % print_every == 0:
                #Validating
                valid_loss = 0
                valid_accuracy = 0

                model.eval()

                with torch.no_grad():
                    for inputs, labels in valid_loader:
                        inputs, labels = inputs.to(device), labels.to(device)

                        logps = model.forward(inputs)
                        current_loss = criterion(logps, labels)
                        valid_loss += current_loss.item()

                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        valid_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

                print(f"Epoch {epoch+1}/{epochs}, "
                      f"Train loss: {training_loss/print_every:.3f}, "
                      f"Valid loss: {valid_loss/len(valid_loader):.3f}, "
                      f"Valid accuracy: {valid_accuracy/len(valid_loader):.3f}"
                      )
                
                training_loss = 0

                model.train()

    print("\nTraining Completed")            
    
    return model, criterion

def test_model(model, test_loader, criterion, gpu):
    print("\ninside test_model")
    
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
    
    test_loss = 0
    test_accuracy = 0
    
    model.eval()
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
            test_loss += batch_loss.item()

            # Accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            test_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    print(
          f"Test loss: {test_loss/len(test_loader):.3f}, "
          f"Test accuracy: {test_accuracy/len(test_loader):.3f}"
          "Testing Done!"
          )
    
def save(model, architecture, hidden_units, epochs, learning_rate, save_dir):
    print("\nInside model save ")
    
    checkpoint = {
        'architecture': architecture,
        'hidden_units': hidden_units,
        'epochs': epochs,
        'learning_rate': learning_rate,
        'model_state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    
    checkpoint_path = save_dir + "/checkpoint.pth"

    torch.save(checkpoint, checkpoint_path)
    
    print("\nCheckpoint filepath: {}".format(checkpoint_path))
    
def load(filepath):
    print("\nInside model load")

    checkpoint = torch.load(filepath)
    model, optimizer = create_model(checkpoint['architecture'], checkpoint['hidden_units'], checkpoint['learning_rate'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.class_to_idx = checkpoint['class_to_idx'] 
    
    return model

def predict(model, processed_image, topk, gpu): 
    
    
    if gpu:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cpu")
        
    print("User choice: {}, running on: {}".format("gpu" if gpu else "cpu", device.type))
    model.to(device)
    processed_image = processed_image.to(device)
    
    model.eval()
    
    with torch.no_grad():
        logps = model.forward(processed_image.unsqueeze(0))
        ps = torch.exp(logps)
        probabilities, labels = ps.topk(topk, dim=1)
        
        class_to_idx_inv = {model.class_to_idx[i]: i for i in model.class_to_idx}
        classes = list()
    
        for label in labels.cpu.numpy()[0]:
            classes.append(class_to_idx_inv[label])
        
        return probabilities.numpy()[0], classes