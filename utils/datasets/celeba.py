import pandas as pd
import os
import torch
from torchvision import datasets
from torchvision import transforms as T
from torch.utils.data._utils.collate import default_collate

cols = [
    '5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes',
    'Bald', 'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
    'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
    'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
    'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
    'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline',
    'Rosy_Cheeks', 'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair',
    'Wearing_Earrings', 'Wearing_Hat', 'Wearing_Lipstick',
    'Wearing_Necklace', 'Wearing_Necktie', 'Young'
]


def get_celeba(batch_size, *, root_dir=os.getcwd()):
    """
    We are only picking a selected number of features from
    CelebA dataset.
    """
    features = [
        "Male",
        "Heavy_Makeup",
        "No_Beard",
        "Mustache",
        "Sideburns",
        "Attractive",
        "Bald",
        "Rosy_Cheeks",
        "Eyeglasses",
        "Smiling"
    ]
    features_to_indicies = [
        cols.index(feature) for feature in features
    ]

    def collate(batch):
        batch = default_collate(batch)
        return [batch[0], batch[1][:, features_to_indicies]]

    def get_loader(split):
        transform = T.Compose([
            T.Resize((256, 256)),
            T.ToTensor(),
        ])
        return torch.utils.data.DataLoader(
                    datasets.CelebA(root_dir, split=split, download=True, transform=transform),
                    batch_size=batch_size,
                    shuffle=True,
                    collate_fn=collate
               )

    return get_loader("train"), get_loader("valid")
