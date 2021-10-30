import argparse
import neptune.new as neptune
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor, Resize, Compose
from dataloader.TextDataset import Text
from model.VBDeblur import VB_Blur
from train.train import train


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--exp_id', type=str, default='exp_0')

    parser.add_argument('--resume', type=int, default=0, help='resume the trained model')
    parser.add_argument('--test', type=int, default=0, help='test with trained model')

    # train
    parser.add_argument('--epochs', type=int, default=20, help='number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
    parser.add_argument('--lr_decay', type=float, default=1e-1, help="learning rate decay after each epochs")

    # loss
    parser.add_argument('--simplex_scale', type=float, default=2e4, help="scale the simplex of blur kernel distribution")
    parser.add_argument('--sigma', type=float, default=1e-2, help="sigma")
    parser.add_argument('--epsilon', type=float, default=1e-2, help="epsilon")

    # model
    parser.add_argument('--res_number', type=int, default=2, help="number of resblock in encoder/decoder block")
    parser.add_argument('--channel_number', type=int, default=2, help="number of channels in encoder/decoder block")
    parser.add_argument('--blur_channel_number', type=int, default=2, help="number of blur channels sft layer in "
                                                                           "encoder/decoder block")
    # seed
    parser.add_argument('--seed', type=int, default=1, help='random seed')

    args = parser.parse_args()
    return args


if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    run = neptune.init(
        project="kevinqd/image-deblur",
        api_token="eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiJhZDYzNWE1NS05NjliLTQ5YjQtYmRhNS0xNTE2NzNlN2E2NjEifQ==",
    )

    img_transform = Compose([
        Resize((112, 112)),
        ToTensor(),
    ])

    blur_transform = Compose([
        Resize((17, 17)),
        ToTensor(),
    ])

    text = Text("/notebooks/dataset/text/data/", img_transform, blur_transform)
    no_train = int(0.8 * len(text))
    no_test = len(text) - no_train
    train_data, test_data = torch.utils.data.random_split(text, [no_train, no_test])
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=True, num_workers=2)

    torch.cuda.empty_cache()
    model = VB_Blur(img_size=112,
                    kernel_size=17,
                    n_res_block=args.res_number,
                    channel=args.channel_number,
                    blur_channel=args.blur_channel_number).to(device)

    model = nn.DataParallel(model).to("cuda")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.lr_decay, verbose=True)

    for t in range(args.epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        hist = train(model,
                     test_dataloader,
                     optimizer,
                     run,
                     exp_id=args.exp_id,
                     sigma=args.sigma,
                     scaling_term=args.simplex_scale,
                     epsilon=args.epsilon)
        scheduler.step()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
