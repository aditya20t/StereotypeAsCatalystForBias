import argparse

def parseArguments():
# Define the parser
    parser = argparse.ArgumentParser(description='Process terminal flags')

    # Add flags
    parser.add_argument('--dataset_1', type=str, help='First Dataset name')
    parser.add_argument('--dataset_2', type=str, help='Second Dataset name')
    parser.add_argument('--model', type=str, help='Model name')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--epochs', type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, help='Learning rate')
    parser.add_argument('--weight_decay', default=1e-4, type=float, help='Weight decay')
    parser.add_argument('--device', default='cuda', type=str, help='Device')
    parser.add_argument('--threshold', default=0.5, type=float, help='Threshold')
    parser.add_argument('--alpha', default=0.5, type=float, help='Weightage to bias')
    parser.add_argument('--task_1', type=str, help='Task 1 name')
    parser.add_argument('--task_2', type=str, help='Task 2 name')

    # Parse the arguments
    args = parser.parse_args()
    return args