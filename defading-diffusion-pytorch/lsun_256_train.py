from defading_diffusion_pytorch import Unet, GaussianDiffusion, Trainer, Model
import torchvision
import os
import errno
import shutil
import argparse


def create_folder(path):
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass


def del_folder(path):
    try:
        shutil.rmtree(path)
    except OSError as exc:
        pass


parser = argparse.ArgumentParser()
parser.add_argument('--time_steps', default=50, type=int)
parser.add_argument('--train_steps', default=700000, type=int)
parser.add_argument('--save_folder', default=None, type=str)
parser.add_argument('--blur_std', default=0.1, type=float)
parser.add_argument('--load_path', default=None, type=str)
parser.add_argument('--data_path', default='./root_cifar10/', type=str)
parser.add_argument('--fade_routine', default='Random_Incremental', type=str)
parser.add_argument('--sampling_routine', default='x0_step_down', type=str)
parser.add_argument('--discrete', action="store_true")
parser.add_argument('--remove_time_embed', action="store_true")
parser.add_argument('--residual', action="store_true")

args = parser.parse_args()
print(args)

model = Model(resolution=32,
              in_channels=3,
              out_ch=3,
              ch=128,
              ch_mult=(1, 2, 2, 2),
              num_res_blocks=2,
              attn_resolutions=(16,),
              dropout=0.1).cuda()

diffusion = GaussianDiffusion(
    model,
    image_size=256,
    device_of_kernel='cuda',
    channels=3,
    timesteps=args.time_steps,
    loss_type='l1',
    kernel_std=args.blur_std,
    fade_routine=args.fade_routine,
    sampling_routine=args.sampling_routine,
    discrete=args.discrete
).cuda()

trainer = Trainer(
    diffusion,
    args.data_path,
    image_size=256,
    train_batch_size=8,
    train_lr=2e-5,
    train_num_steps=args.train_steps,
    gradient_accumulate_every=2,
    ema_decay=0.995,
    fp16=False,
    results_folder=args.save_folder,
    load_path=args.load_path,
    dataset='train'
)

trainer.train()
