import torch
import timm


width = 64

def Selec_embedding(model_name, **params):
    if model_name == 'Resnet_18':
        model = Resnet_Encoder(**params)
    elif model_name == 'Vgg_11':
        model = VGG_Encoder(**params)
    elif model_name == 'ViT':
        model = ViT_Encoder(**params)
    else:
        raise Exception('No encoder selected')
    return model

def Selec_model_two_classes(model_name, **params):
    if model_name == 'Resnet_18':
        model = ResNet_TwoClass(**params)
    elif model_name == 'Vgg_11':
        model = VGG_TwoClass(**params)
    elif model_name == 'ViT':
        model = ViT_TwoClass(**params)
    return model

class ViT_TwoClass(torch.nn.Module):

    def __init__(self, channels=3, freeze=True):
        super(ViT_TwoClass, self).__init__()

        self.ViT = timm.create_model('vit_base_patch16_224', pretrained=True, in_chans=channels, num_classes=0)
        
        for param in self.ViT.parameters():
            param.requires_grad = not freeze
        
        self.activation = torch.nn.ReLU()
        self.linear1 = torch.nn.Linear(self.ViT.num_features, 2*width)
        self.output = torch.nn.Linear(2*width, 2)
        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=0.4)

    def forward(self, x):
        x = self.ViT(x)
        x = self.linear1(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.output(x) 
        return x

class ResNet_TwoClass(torch.nn.Module):

    def __init__(self,num_layers=18, channels=3, freeze=True):
        super(ResNet_TwoClass, self).__init__()

        if num_layers == 18:
            Rnet = timm.create_model('resnet18', pretrained=True, in_chans=channels, num_classes=0, global_pool='avg')
        elif num_layers == 50:
            Rnet = timm.create_model('resnet50', pretrained=True, in_chans=channels, num_classes=0, global_pool='avg')
        
        for param in Rnet.parameters():
            param.requires_grad = not freeze
        
        self.Rnet = Rnet
        self.Rnet.fc = torch.nn.Sequential(
            torch.nn.Linear(self.Rnet.num_features, 2*width),
            torch.nn.Dropout(p=0.6),
            torch.nn.ReLU(),
            torch.nn.Linear(2*width, 2)
        )

    def forward(self, x):
        x = self.Rnet(x)
        return x

class VGG_TwoClass(torch.nn.Module):

    def __init__(self,num_layers=11, channels=3, freeze=True):
        super(VGG_TwoClass, self).__init__()

        if num_layers == 11:
            Rnet = timm.create_model('vgg11_bn', pretrained=True, in_chans=channels, num_classes=0, global_pool='avg')
        elif num_layers == 13:
            Rnet = timm.create_model('vgg13_bn', pretrained=True, in_chans=channels, num_classes=0, global_pool='avg')
        elif num_layers == 16:
            Rnet = timm.create_model('vgg16_bn', pretrained=True, in_chans=channels, num_classes=0, global_pool='avg')
        elif num_layers == 19:
            Rnet = timm.create_model('vgg19_bn', pretrained=True, in_chans=channels, num_classes=0, global_pool='avg')
        
        for param in Rnet.parameters():
            param.requires_grad = not freeze
        
        self.Rnet = Rnet
        self.Rnet.head.fc = torch.nn.Sequential(
            torch.nn.Linear(self.Rnet.num_features, 2*width),
            torch.nn.Dropout(p=0.6),
            torch.nn.ReLU(),
            torch.nn.Linear(2*width, 2)
        )

    def forward(self, x):
        x = self.Rnet(x)
        return x

class ViT_Encoder(torch.nn.Module):

    def __init__(self, channels=3, freeze=True):
        super(ViT_Encoder, self).__init__()

        self.ViT = timm.create_model('vit_base_patch16_224', pretrained=True, in_chans=channels, num_classes=0)
        self.linear1 = torch.nn.Linear(self.ViT.num_features, 2*width)
        self.relu = torch.nn.ReLU()

        for param in self.ViT.parameters():
            param.requires_grad = not freeze

        self.embedding = torch.nn.Sequential(
            self.ViT,
            self.linear1,
            self.relu
        )

        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=2*width, out_features=width),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=width, out_features=width)
        )

        self.linear_probe = torch.nn.Sequential(
            torch.nn.Linear(in_features=2*width, out_features=2)
        )
    
    def calculate_embedding(self, image):
        return self.embedding(image)

    def calculate_linear_probe(self, x):
        x = self.embedding(x)
        return self.linear_probe(x)

    def forward(self, x):
        
        embedding = self.embedding(x)
        
        projection = self.projection_head(embedding)

        return projection

class Resnet_Encoder(torch.nn.Module):

    def __init__(self,num_layers=18, channels=3, freeze=True):
        super(Resnet_Encoder, self).__init__()

        if num_layers == 18:
            self.Rnet = timm.create_model('resnet18', pretrained=True, in_chans=channels, num_classes=0)
        elif num_layers == 50:
            self.Rnet = timm.create_model('resnet50', pretrained=True, in_chans=channels, num_classes=0)
        self.linear1 = torch.nn.Linear(self.Rnet.num_features, 2*width)
        self.relu = torch.nn.ReLU()

        for param in self.Rnet.parameters():
            param.requires_grad = not freeze

        self.embedding = torch.nn.Sequential(
            self.Rnet,
            self.linear1,
            self.relu
        )

        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=2*width, out_features=width),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=width, out_features=width)
        )

        self.linear_probe = torch.nn.Sequential(
            torch.nn.Linear(in_features=2*width, out_features=2)
        )
    
    def calculate_embedding(self, image):
        return self.embedding(image)

    def calculate_linear_probe(self, x):
        x = self.embedding(x)
        return self.linear_probe(x)

    def forward(self, x):
        
        embedding = self.embedding(x)
        
        projection = self.projection_head(embedding)

        return projection

class VGG_Encoder(torch.nn.Module):

    def __init__(self,num_layers=11, channels=3, freeze=True):
        super(VGG_Encoder, self).__init__()

        if num_layers == 11:
            VGG = timm.create_model('vgg11_bn', pretrained=True, in_chans=channels, num_classes=0)
        elif num_layers == 13:
            VGG = timm.create_model('vgg13_bn', pretrained=True, in_chans=channels, num_classes=0)
        elif num_layers == 16:
            VGG = timm.create_model('vgg16_bn', pretrained=True, in_chans=channels, num_classes=0)
        elif num_layers == 19:
            VGG = timm.create_model('vgg19_bn', pretrained=True, in_chans=channels, num_classes=0)

        for param in VGG.parameters():
            param.requires_grad = not freeze

        VGG.head.fc = torch.nn.Sequential(
            torch.nn.Linear(VGG.num_features, 2*width),
            torch.nn.ReLU(),
            )

        self.embedding = VGG

        self.projection_head = torch.nn.Sequential(
            torch.nn.Linear(in_features=2*width, out_features=width),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=width, out_features=width)
        )

        self.linear_probe = torch.nn.Sequential(
            torch.nn.Linear(in_features=2*width, out_features=2)
        )
    
    def calculate_embedding(self, image):
        return self.embedding(image)

    def calculate_linear_probe(self, x):
        x = self.embedding(x)
        return self.linear_probe(x)

    def forward(self, x):
        
        embedding = self.embedding(x)
        
        projection = self.projection_head(embedding)

        return projection

def finetuning_model(embedding_input, freeze=True):
    class final_model(torch.nn.Module):
        def __init__(self,embedding, freeze=True):
            super(final_model, self).__init__()

            for param in embedding.parameters():
                param.requires_grad = not freeze

            self.embedding = embedding
            self.linear = torch.nn.Linear(2*width, width)
            self.output = torch.nn.Linear(width, 2)
            self.relu = torch.nn.ReLU()
            self.dropout = torch.nn.Dropout(p=0.4)

        def forward_emb(self,x):
            return self.embedding(x)

        def forward(self, x):
            x = self.embedding(x)
            x = self.linear(x)
            x = self.relu(x)
            x = self.dropout(x)
            x = self.output(x) 
            return x
    return final_model(embedding_input, freeze=freeze)