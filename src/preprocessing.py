from torchvision import transforms


train_transform = transforms.Compose([
    # rescaling
    transforms.Resize((64, 64)),
    #converting to grey scale
    transforms.Grayscale(num_output_channels=1), 
    #simple flipping and rotating for data augmentation, as the dataset size is small
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    # since training from scratch then no need for the [-1 1] normalization, [0 1] is enough
    transforms.ToTensor()
    ])

#should not do any data augmentation on validation set

test_transform = transforms.Compose([
    # rescaling
    transforms.Resize((64, 64)),
    #converting to grey scale
    transforms.Grayscale(num_output_channels=1), 
    transforms.ToTensor()
])