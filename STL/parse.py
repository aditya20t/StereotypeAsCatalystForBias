import argparse

def parseArguments():
# Define the parser
    parser = argparse.ArgumentParser(description='Process terminal flags')

    # Add flags
    parser.add_argument('--dataset', type=str, help='Dataset name')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--device', default='cuda', type=str, help='Device')
    parser.add_argument('--threshold', default=0.5, type=float, help='Threshold')

    # Parse the arguments
    args = parser.parse_args()
    return args