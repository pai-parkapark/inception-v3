from dataprovider.data_setter import CIFAR10DataSetter
from dataprovider.data_loader import CIFAR10DataLoader
from torch.utils.data import Subset
import torchvision.transforms as transforms
import torch.optim as optim
from model.inceptionv3.loss import inception_v3_loss
from model.inceptionv3.metric import inception_v3_accuracy
from model.inceptionv3.model import InceptionV3
from trainer import InceptionTrainer


def train(model, loss_fn, metric_fn, epochs=20, batch_size=64, num_classes=10, lr=0.001):
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_data_setter = CIFAR10DataSetter(root='./data', train=True, download=False, transform=transform)
    train_subset_indices = list(range(batch_size * 40))
    sub_train_dataset = Subset(train_data_setter, train_subset_indices)
    train_data_loader = CIFAR10DataLoader(sub_train_dataset, batch_size=batch_size, shuffle=True)

    valid_data_setter = CIFAR10DataSetter(root='./data', train=False, download=False, transform=transform)
    valid_subset_indices = list(range(batch_size * 10))
    sub_valid_dataset = Subset(valid_data_setter, valid_subset_indices)
    valid_data_loader = CIFAR10DataLoader(sub_valid_dataset, batch_size=batch_size, shuffle=True)

    if model == 'inception_v3':
        model = InceptionV3(num_classes=num_classes, aux_logits=True)

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if loss_fn == 'inception_v3':
        loss_fn = inception_v3_loss

    if metric_fn == 'inception_v3':
        metric_fn = inception_v3_accuracy

    inception_v3_trainer = InceptionTrainer(model=model, loss=loss_fn, optimizer=optimizer, metric=metric_fn,
                                            train_data_loader=train_data_loader, valid_data_loader=valid_data_loader,
                                            mac_gpu=True)
    train_loss, train_acc = inception_v3_trainer.train(epochs)
    print(f"Trian Loss : {train_loss}, Trian accuracy : {train_acc}")

    val_loss, val_acc = inception_v3_trainer.validate()
    print(f"Validation Loss : {val_loss}, Validation accuracy : {val_acc}")


if __name__ == '__main__':
    train(model="inception_v3",
          loss_fn="inception_v3",
          metric_fn="inception_v3",
          epochs=2,
          batch_size=64,
          num_classes=10,
          lr=0.001)
