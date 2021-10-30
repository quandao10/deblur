import os
import torch


def resume(name, model, optimizer):
    checkpoint_path = './checkpoints/{}.pth'.format(name)
    assert os.path.exists(checkpoint_path), ('checkpoint do not exits for %s' % checkpoint_path)

    checkpoint_saved = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint_saved['model_state_dict'])
    optimizer.load_state_dict(checkpoint_saved['optimizer_state_dict'])

    print('Resume completed for the model\n')

    return model, optimizer


def predict(model, data, path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    data = data.to(device)
    mu, log_var, simplex, blur_kernel, latent_img = model(data)
    return None
