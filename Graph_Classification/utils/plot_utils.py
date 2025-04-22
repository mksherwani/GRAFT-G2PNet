import matplotlib.pyplot as plt
import os

def plot_training_curves(train_losses, val_losses, train_acc=None, val_acc=None, save_path='outputs/curve.png'):
    fig, axs = plt.subplots(2 if train_acc else 1, 1, figsize=(10, 6))

    if train_acc:
        axs[0].plot(train_losses, label='Train Loss')
        axs[0].plot(val_losses, label='Val Loss')
        axs[0].legend()
        axs[0].set_title("Loss")

        axs[1].plot(train_acc, label='Train Accuracy')
        axs[1].plot(val_acc, label='Val Accuracy')
        axs[1].legend()
        axs[1].set_title("Accuracy")
    else:
        axs.plot(train_losses, label='Train Loss')
        axs.plot(val_losses, label='Val Loss')
        axs.set_title("Loss Only")
        axs.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()
