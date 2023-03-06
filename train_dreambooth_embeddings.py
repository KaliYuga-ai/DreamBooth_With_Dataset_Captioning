import argparse
import hashlib
import itertools
import random
import json
import logging
import math
import os
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.utils.data import Dataset

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, DDIMScheduler, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from diffusers.utils.import_utils import is_xformers_available
from huggingface_hub import HfFolder, Repository, whoami
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
from transformers import CLIPTextModel, CLIPTokenizer


torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)

class UNet2DConditionModel(nn.Module):
    def __init__(
        self,
        num_steps,
        cond_dim,
        image_size,
        embedding_dim,
        num_channels=3,
        use_checkpoint=False,
        use_fp16=False,
    ):
        super().__init__()

        self.num_steps = num_steps
        self.image_size = image_size
        self.num_channels = num_channels
        self.use_checkpoint = use_checkpoint
        self.use_fp16 = use_fp16

        self.cond_encoder = nn.Sequential(
            nn.Linear(cond_dim + embedding_dim, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 256),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.encoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(self.num_channels + (i > 0) * self.num_channels * 2, 32, 3, stride=1, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(32, 32, 3, stride=1, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                for i in range(num_steps)
            ]
        )

        self.center = nn.Sequential(
            nn.Conv2d(self.num_channels * 2 ** (num_steps - 1), 32 * 2 ** (num_steps - 1), 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32 * 2 ** (num_steps - 1), 32 * 2 ** (num_steps - 1), 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32 * 2 ** (num_steps - 1), 32 * 2 ** (num_steps - 2), 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(32 * 2 ** (num_steps - i), 32 * 2 ** (num_steps - i - 1), 3, stride=1, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.Conv2d(32 * 2 ** (num_steps - i - 1), 32 * 2 ** (num_steps - i - 1), 3, stride=1, padding=1),
                    nn.LeakyReLU(0.2, inplace=True),
                    nn.ConvTranspose2d(
                        32 * 2 ** (num_steps - i - 1),
                        32 * 2 ** (num_steps - i - 2),
                        4,
                        stride=2,
                        padding=1,
                    ),
                    nn.LeakyReLU(0.2, inplace=True),
                )
                for i in range(num_steps)
            ]
        )

# Define the train function
def train(model, optimizer, train_loader, text_encoder, scheduler, global_step, args):
    model.train()
    text_encoder.train()

    for i, batch in enumerate(train_loader):
        image = batch["image"].to(args.device)
        embedding = batch["embedding"].to(args.device)
        text = batch["text"].to(args.device)

        # Generate text embeddings
        text_input = text[:, :-1]
        text_target = text[:, 1:]
        text_mask = torch.BoolTensor(text_input != text_encoder.tokenizer.pad_token_id).to(args.device)
        text_embedding = text_encoder(text_input, text_mask)

        # Combine image and text embeddings
        input_embedding = torch.cat([embedding.unsqueeze(0), text_embedding], dim=1)

        # Generate samples and calculate loss
        loss = model(input_embedding, image, text_target)

        # Backward pass and optimization step
        loss.backward()
        if args.gradient_accumulation_steps > 1:
            if (i + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad()

        # Learning rate scheduling
        if args.lr_scheduler == "constant":
            pass
        elif args.lr_scheduler == "warmup":
            scheduler.step()
        elif args.lr_scheduler == "reduce_on_plateau":
            scheduler.step(loss)
        else:
            raise ValueError(f"Invalid learning rate scheduler '{args.lr_scheduler}'")

        # Log training progress
        if i % args.log_interval == 0:
            logging.info(f"Step {global_step}: loss = {loss.item()}")

def train_step(text_embeddings, image_embeddings, model, optimizer, diffusion, loss_fn, scheduler=None):
    model.train()

    # Sample noise input for each diffusion step
    z = diffusion.sample_z(batch_size, device=device)

    # Iterate over the diffusion steps in reverse order
    for i in range(diffusion.num_steps - 1, -1, -1):
        # Generate the output image for this diffusion step
        output_image = model(text_embeddings, image_embeddings, z[i], i)

        # Calculate the loss between the generated image and the ground truth image
        loss = loss_fn(output_image, target_image, i)

        # Update the model parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Update the learning rate scheduler
    if scheduler is not None:
        scheduler.step()

    return loss.item()

def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--pretrained_vae_name_or_path",
        type=str,
        default=None,
        help="Path to pretrained vae or vae identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--tokenizer_name",
        type=str,
        default=None,
        help="Pretrained tokenizer name or path if not the same as model_name",
    )
    parser.add_argument(
        "--instance_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of instance images.",
    )
    parser.add_argument(
        "--class_data_dir",
        type=str,
        default=None,
        help="A folder containing the training data of class images.",
    )
    parser.add_argument(
        "--instance_prompt",
        type=str,
        default=None,
        help="The prompt with identifier specifying the instance",
    )
    parser.add_argument(
        "--class_prompt",
        type=str,
        default=None,
        help="The prompt to specify images in the same class as provided instance images.",
    )
    parser.add_argument(
        "--save_sample_prompt",
        type=str,
        default=None,
        help="The prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--save_sample_negative_prompt",
        type=str,
        default=None,
        help="The negative prompt used to generate sample outputs to save.",
    )
    parser.add_argument(
        "--n_save_sample",
        type=int,
        default=4,
        help="The number of samples to save.",
    )
    parser.add_argument(
        "--save_guidance_scale",
        type=float,
        default=7.5,
        help="CFG for save sample.",
    )
    parser.add_argument(
        "--save_infer_steps",
        type=int,
        default=20,
        help="The number of inference steps for save sample.",
    )
    parser.add_argument(
        "--pad_tokens",
        default=False,
        action="store_true",
        help="Flag to pad tokens to length 77.",
    )
    parser.add_argument(
        "--with_prior_preservation",
        default=False,
        action="store_true",
        help="Flag to add prior preservation loss.",
    )
    parser.add_argument("--prior_loss_weight", type=float, default=1.0, help="The weight of prior preservation loss.")
    parser.add_argument(
        "--num_class_images",
        type=int,
        default=100,
        help=(
            "Minimal class images for prior preservation loss. If not have enough images, additional images will be"
            " sampled with class_prompt."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="text-inversion-model",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=512,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop", action="store_true", help="Whether to center crop images before resizing to resolution"
    )
    parser.add_argument("--train_text_encoder", action="store_true", help="Whether to train the text encoder")
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument(
        "--sample_batch_size", type=int, default=4, help="Batch size (per device) for sampling images."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N steps.")
    parser.add_argument("--save_interval", type=int, default=10_000, help="Save weights every N steps.")
    parser.add_argument("--save_min_steps", type=int, default=0, help="Start saving weights after N steps.")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument("--not_cache_latents", action="store_true", help="Do not precompute and cache latents from VAE.")
    parser.add_argument("--hflip", action="store_true", help="Apply horizontal flip data augmentation.")
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
    parser.add_argument(
        "--concepts_list",
        type=str,
        default=None,
        help="Path to json containing multiple concepts, will overwrite parameters like instance_prompt, class_prompt, etc.",
    )
    parser.add_argument(
        "--read_prompts_from_txts",
        action="store_true",
        help="Use prompt per image. Put prompts in the same directory as images, e.g. for image.png create image.png.txt.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    return args


class DreamBoothDataset(Dataset):
    def __init__(self, instance_data_dir, instance_prompt, embeddings_dir, embedding_model, transform=None):
        self.instance_data_dir = instance_data_dir
        self.instance_prompt = instance_prompt
        self.embeddings_dir = embeddings_dir
        self.embedding_model = embedding_model
        self.transform = transform
        
        # Load image embeddings from file
        embeddings_file = os.path.join(embeddings_dir, "embeddings.npy")
        self.embeddings = np.load(embeddings_file)

        # Load instance prompts from file
        with open(instance_prompt) as f:
            self.instance_prompts = f.readlines()
        self.instance_prompts = [x.strip() for x in self.instance_prompts]

    def __len__(self):
        return len(self.instance_prompts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # Load image and preprocess
        img_path = os.path.join(self.instance_data_dir, f"{idx}.png")
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        # Get image embedding
        embedding = self.embeddings[idx]

        # Get instance prompt
        instance_prompt = self.instance_prompts[idx]

        sample = {"image": image, "embedding": embedding, "instance_prompt": instance_prompt}

        return sample

class PromptDataset(Dataset):
    "A simple dataset to prepare the prompts to generate class images on multiple GPUs."

    def __init__(self, prompt, num_samples):
        self.prompt = prompt
        self.num_samples = num_samples

    def __len__(self):
        return self.num_samples

    def __getitem__(self, index):
        example = {}
        example["prompt"] = self.prompt
        example["index"] = index
        return example


class LatentsDataset(Dataset):
    def __init__(self, latents_cache, text_encoder_cache):
        self.latents_cache = latents_cache
        self.text_encoder_cache = text_encoder_cache

    def __len__(self):
        return len(self.latents_cache)

    def __getitem__(self, index):
        return self.latents_cache[index], self.text_encoder_cache[index]


class AverageMeter:
    def __init__(self, name=None):
        self.name = name
        self.reset()

    def reset(self):
        self.sum = self.count = self.avg = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def get_full_repo_name(model_id: str, organization: Optional[str] = None, token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def main(args):
    accelerator = Accelerator()
    set_seed(args.seed)

    # Load text encoder
    text_encoder = CLIPTextModel.from_pretrained(args.clip_model_name, cache_dir=args.cache_dir)
    text_encoder.eval().to(args.device)

    # Load DreamBooth dataset
    train_dataset = DreamBoothDataset(
        instance_data_dir=args.instance_data_dir,
        instance_prompt=args.instance_prompt,
        embeddings_dir=args.embeddings_dir,
        embedding_model=args.embedding_model,
        transform=transforms.Compose([
            transforms.Resize(args.image_size),
            transforms.CenterCrop(args.image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.5] * 3, [0.5] * 3),
        ])
    )

    # Create data loader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )

    # Define model
    diffusion = StableDiffusionPipeline(args.diffusion_steps, args.diffusion_timestep)
    model = UNet2DConditionModel(
        diffusion.num_steps,
        args.embedding_dim + text_encoder.config.hidden_size,
        args.image_size,
        num_channels=args.num_channels,
        use_checkpoint=args.use_checkpoint,
        use_fp16=accelerator.state.is_fp16_available,
    ).to(args.device)
    model = accelerator.prepare(model)

    # Define optimizer and learning rate scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = get_scheduler(
        optimizer,
        args.lr_scheduler,
        args.lr_warmup_epochs,
        args.lr_reduce_patience,
        args.lr_reduce_factor,
        args.lr_min_learning_rate,
    )

    # Define loss function
    loss_fn = AutoencoderKL(
        diffusion,
        args.embedding_dim + text_encoder.config.hidden_size,
        num_channels=args.num_channels,
        use_fp16=accelerator.state.is_fp16_available,
    ).to(args.device)

    # Train model
    global_step = 0
    for epoch in range(args.num_epochs):
        for batch in train_loader:
            batch = accelerator.prepare(batch)
            text_embeddings = text_encoder(batch["text"][:, :-1], attention_mask=batch["text_mask"][:, :-1]).last_hidden_state
            image_embeddings = batch["embedding"]
            loss = train_step(
                text_embeddings,
                image_embeddings,
                model,
                optimizer,
                diffusion,
                loss_fn,
                scheduler,
            )

            # Log training progress
            if global_step % args.log_interval == 0:
                logger.info(f"Epoch {epoch} step {global_step}: loss = {loss}")

            global_step += 1

        # Save checkpoint after each epoch
        checkpoint_dir = args.checkpoint_dir / f"epoch_{epoch}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        accelerator.save(
            {
                "epoch": epoch,
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            checkpoint_dir / "checkpoint.pt",
        )
        
    def collate_fn(examples):
        input_ids = [example["instance_prompt_ids"] for example in examples]
        pixel_values = [example["instance_images"] for example in examples]

        # Concat class and instance examples for prior preservation.
        # We do this to avoid doing two forward passes.
        if args.with_prior_preservation:
            input_ids += [example["class_prompt_ids"] for example in examples]
            pixel_values += [example["class_images"] for example in examples]

        pixel_values = torch.stack(pixel_values)
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

        input_ids = tokenizer.pad(
            {"input_ids": input_ids},
            padding=True,
            return_tensors="pt",
        ).input_ids

        batch = {
            "input_ids": input_ids,
            "pixel_values": pixel_values,
        }
        return batch

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, pin_memory=True
    )

    weight_dtype = torch.float32
    if args.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif args.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu.
    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    vae.to(accelerator.device, dtype=weight_dtype)
    if not args.train_text_encoder:
        text_encoder.to(accelerator.device, dtype=weight_dtype)

    if not args.not_cache_latents:
        latents_cache = []
        text_encoder_cache = []
        for batch in tqdm(train_dataloader, desc="Caching latents"):
            with torch.no_grad():
                batch["pixel_values"] = batch["pixel_values"].to(accelerator.device, non_blocking=True, dtype=weight_dtype)
                batch["input_ids"] = batch["input_ids"].to(accelerator.device, non_blocking=True)
                latents_cache.append(vae.encode(batch["pixel_values"]).latent_dist)
                if args.train_text_encoder:
                    text_encoder_cache.append(batch["input_ids"])
                else:
                    text_encoder_cache.append(text_encoder(batch["input_ids"])[0])
        train_dataset = LatentsDataset(latents_cache, text_encoder_cache)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=1, collate_fn=lambda x: x, shuffle=True)

        del vae
        if not args.train_text_encoder:
            del text_encoder
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        overrode_max_train_steps = True

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * args.gradient_accumulation_steps,
        num_training_steps=args.max_train_steps * args.gradient_accumulation_steps,
    )

    if args.train_text_encoder:
        unet, text_encoder, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, text_encoder, optimizer, train_dataloader, lr_scheduler
        )
    else:
        unet, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
            unet, optimizer, train_dataloader, lr_scheduler
        )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if overrode_max_train_steps:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("dreambooth")

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    def save_weights(step):
        # Create the pipeline using using the trained modules and save it.
        if accelerator.is_main_process:
            if args.train_text_encoder:
                text_enc_model = accelerator.unwrap_model(text_encoder, keep_fp32_wrapper=True)
            else:
                text_enc_model = CLIPTextModel.from_pretrained(args.pretrained_model_name_or_path, subfolder="text_encoder", revision=args.revision)
            pipeline = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model_name_or_path,
                unet=accelerator.unwrap_model(unet, keep_fp32_wrapper=True),
                text_encoder=text_enc_model,
                vae=AutoencoderKL.from_pretrained(
                    args.pretrained_vae_name_or_path or args.pretrained_model_name_or_path,
                    subfolder=None if args.pretrained_vae_name_or_path else "vae",
                    revision=None if args.pretrained_vae_name_or_path else args.revision,
                ),
                safety_checker=None,
                torch_dtype=torch.float16,
                revision=args.revision,
            )
            pipeline.scheduler = DDIMScheduler.from_config(pipeline.scheduler.config)
            if is_xformers_available():
                pipeline.enable_xformers_memory_efficient_attention()
            save_dir = os.path.join(args.output_dir, f"{step}")
            pipeline.save_pretrained(save_dir)
            with open(os.path.join(save_dir, "args.json"), "w") as f:
                json.dump(args.__dict__, f, indent=2)

            if args.save_sample_prompt is not None:
                pipeline = pipeline.to(accelerator.device)
                g_cuda = torch.Generator(device=accelerator.device).manual_seed(args.seed)
                pipeline.set_progress_bar_config(disable=True)
                sample_dir = os.path.join(save_dir, "samples")
                os.makedirs(sample_dir, exist_ok=True)
                with torch.autocast("cuda"), torch.inference_mode():
                    for i in tqdm(range(args.n_save_sample), desc="Generating samples"):
                        images = pipeline(
                            args.save_sample_prompt,
                            negative_prompt=args.save_sample_negative_prompt,
                            guidance_scale=args.save_guidance_scale,
                            num_inference_steps=args.save_infer_steps,
                            generator=g_cuda
                        ).images
                        images[0].save(os.path.join(sample_dir, f"{i}.png"))
                del pipeline
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            print(f"[*] Weights saved at {save_dir}")

    # Only show the progress bar once on each machine.
    progress_bar = tqdm(range(args.max_train_steps), disable=not accelerator.is_local_main_process)
    progress_bar.set_description("Steps")
    global_step = 0
    loss_avg = AverageMeter()
    text_enc_context = nullcontext() if args.train_text_encoder else torch.no_grad()
    for epoch in range(args.num_train_epochs):
        unet.train()
        if args.train_text_encoder:
            text_encoder.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                with torch.no_grad():
                    if not args.not_cache_latents:
                        latent_dist = batch[0][0]
                    else:
                        latent_dist = vae.encode(batch["pixel_values"].to(dtype=weight_dtype)).latent_dist
                    latents = latent_dist.sample() * 0.18215

                # Sample noise that we'll add to the latents
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                # Sample a random timestep for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=latents.device)
                timesteps = timesteps.long()

                # Add noise to the latents according to the noise magnitude at each timestep
                # (this is the forward diffusion process)
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)

                # Get the text embedding for conditioning
                with text_enc_context:
                    if not args.not_cache_latents:
                        if args.train_text_encoder:
                            encoder_hidden_states = text_encoder(batch[0][1])[0]
                        else:
                            encoder_hidden_states = batch[0][1]
                    else:
                        encoder_hidden_states = text_encoder(batch["input_ids"])[0]

                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample

                # Get the target for loss depending on the prediction type
                if noise_scheduler.config.prediction_type == "epsilon":
                    target = noise
                elif noise_scheduler.config.prediction_type == "v_prediction":
                    target = noise_scheduler.get_velocity(latents, noise, timesteps)
                else:
                    raise ValueError(f"Unknown prediction type {noise_scheduler.config.prediction_type}")

                if args.with_prior_preservation:
                    # Chunk the noise and model_pred into two parts and compute the loss on each part separately.
                    model_pred, model_pred_prior = torch.chunk(model_pred, 2, dim=0)
                    target, target_prior = torch.chunk(target, 2, dim=0)

                    # Compute instance loss
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                    # Compute prior loss
                    prior_loss = F.mse_loss(model_pred_prior.float(), target_prior.float(), reduction="mean")

                    # Add the prior loss to the instance loss.
                    loss = loss + args.prior_loss_weight * prior_loss
                else:
                    loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")

                accelerator.backward(loss)
                # if accelerator.sync_gradients:
                #     params_to_clip = (
                #         itertools.chain(unet.parameters(), text_encoder.parameters())
                #         if args.train_text_encoder
                #         else unet.parameters()
                #     )
                #     accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                loss_avg.update(loss.detach_(), bsz)

            if not global_step % args.log_interval:
                logs = {"loss": loss_avg.avg.item(), "lr": lr_scheduler.get_last_lr()[0]}
                progress_bar.set_postfix(**logs)
                accelerator.log(logs, step=global_step)

            if global_step > 0 and not global_step % args.save_interval and global_step >= args.save_min_steps:
                save_weights(global_step)

            progress_bar.update(1)
            global_step += 1

            if global_step >= args.max_train_steps:
                break

        accelerator.wait_for_everyone()

    save_weights(global_step)

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)

