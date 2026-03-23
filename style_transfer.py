import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import copy

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMAGE_SIZE = 256

def load_image(image, size=IMAGE_SIZE):
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0)
    return image.to(DEVICE, torch.float)

def tensor_to_image(tensor):
    tensor = tensor.squeeze(0).clamp(0, 1)
    return transforms.ToPILImage()(tensor.cpu())

class ContentLoss(nn.Module):
    def __init__(self, target):
        super().__init__()
        self.target = target.detach()
        self.loss = torch.tensor(0.0, device=target.device)

    def forward(self, x):
        self.loss = nn.functional.mse_loss(x, self.target)
        return x

class StyleLoss(nn.Module):
    def __init__(self, target_feature):
        super().__init__()
        self.target = self._gram_matrix(target_feature).detach()
        self.loss = torch.tensor(0.0, device=target_feature.device)

    def _gram_matrix(self, x):
        b, c, h, w = x.size()
        features = x.view(b * c, h * w)
        G = torch.mm(features, features.t())
        return G.div(b * c * h * w)

    def forward(self, x):
        G = self._gram_matrix(x)
        self.loss = nn.functional.mse_loss(G, self.target)
        return x

class Normalization(nn.Module):
    def __init__(self):
        super().__init__()
        mean = torch.tensor([0.485, 0.456, 0.406]).to(DEVICE).view(-1, 1, 1)
        std  = torch.tensor([0.229, 0.224, 0.225]).to(DEVICE).view(-1, 1, 1)
        self.mean = mean
        self.std  = std

    def forward(self, x):
        return (x - self.mean) / self.std

def build_model(cnn, content_img, style_img):
    content_layers = ['conv_4']
    style_layers   = ['conv_1','conv_2','conv_3','conv_4','conv_5']

    normalization = Normalization().to(DEVICE)
    content_losses, style_losses = [], []

    model = nn.Sequential(normalization)
    i = 0
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = f'conv_{i}'
        elif isinstance(layer, nn.ReLU):
            name = f'relu_{i}'
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = f'pool_{i}'
        elif isinstance(layer, nn.BatchNorm2d):
            name = f'bn_{i}'
        else:
            continue

        model.add_module(name, layer)

        if name in content_layers:
            target = model(content_img).detach()
            cl = ContentLoss(target)
            model.add_module(f"content_loss_{i}", cl)
            content_losses.append(cl)

        if name in style_layers:
            target = model(style_img).detach()
            sl = StyleLoss(target)
            model.add_module(f"style_loss_{i}", sl)
            style_losses.append(sl)

    # Trim layers after last loss
    for j in range(len(model) - 1, -1, -1):
        if isinstance(model[j], (ContentLoss, StyleLoss)):
            break
    model = model[:j+1]

    return model, content_losses, style_losses

def run_style_transfer(content_img, style_img, num_steps=300,
                       style_weight=1e6, content_weight=1):
    cnn = models.vgg19(pretrained=True).features.to(DEVICE).eval()
    for p in cnn.parameters():
        p.requires_grad_(False)

    content_tensor = load_image(content_img)
    style_tensor   = load_image(style_img)
    input_img      = content_tensor.clone().requires_grad_(True)

    model, content_losses, style_losses = build_model(cnn, content_tensor, style_tensor)

    optimizer = optim.LBFGS([input_img])
    run = [0]

    while run[0] <= num_steps:
        def closure():
            with torch.no_grad():
                input_img.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            s_loss = sum((sl.loss for sl in style_losses), torch.tensor(0.0, device=DEVICE)) * style_weight
            c_loss = sum((cl.loss for cl in content_losses), torch.tensor(0.0, device=DEVICE)) * content_weight
            loss = s_loss + c_loss
            loss.backward()
            run[0] += 1
            return loss
        optimizer.step(closure)

    with torch.no_grad():
        input_img.clamp_(0, 1)

    return tensor_to_image(input_img)