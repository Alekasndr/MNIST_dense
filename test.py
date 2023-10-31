from train import *

if __name__ == '__main__':
    loaded_model = NeuralNetwork().to(device)
    loaded_model.load_state_dict(torch.load("model.pth"))
    loaded_model.eval()
    loss_and_accuracy_check(test_dataloader, loaded_model, loss_fn)
