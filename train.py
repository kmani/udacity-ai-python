import argparse
from Data_Processing_Helper import create_dataloaders
import Model_Helper

parser = argparse.ArgumentParser()
parser.add_argument('data_directory', help='Filepath to data sets')
parser.add_argument('--save_dir', help='Filepath to store checkpoint.pth')
parser.add_argument('--arch', help='Architecture of model. Allowed values: vgg16, vgg13, resnet50')
parser.add_argument('--learning_rate', help='Learning rate')
parser.add_argument('--hidden_units', help='No. of hidden units')
parser.add_argument('--epochs', help='No. of epochs')
parser.add_argument('--gpu', help='GPU Use', action='store_true')

args = parser.parse_args()

print("Parsing arguments:")

network_architecture = 'vgg16' if args.arch is None else args.arch
learning_rate = 0.001 if args.learning_rate is None else float(args.learning_rate)
hidden_units = 512 if args.hidden_units is None else int(args.hidden_units)

epochs = 4 if args.epochs is None else int(args.epochs)

save_dir = './' if args.save_dir is None else args.save_dir


train_data, train_loader, valid_loader, test_loader = create_dataloaders(args.data_directory)


model, optimizer = Model_Helper.create_model(network_architecture, hidden_units, learning_rate)

model, criterion = Model_Helper.train_model(model, optimizer, epochs, train_loader, valid_loader, args.gpu)

Model_Helper.test_model(model, test_loader, criterion, args.gpu)

model.class_to_idx = train_data.class_to_idx
Model_Helper.save(model, network_architecture, hidden_units, epochs, learning_rate, save_dir)