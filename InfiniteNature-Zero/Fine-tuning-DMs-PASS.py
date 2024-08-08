import os
import sys

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from datasets import load_dataset
from diffusers import DDIMScheduler, DDPMPipeline
from matplotlib import pyplot as plt
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import safetensors
import wandb


def fine_tuning_example_v01():
    # 选择设备
    device = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练模型并设置设备
    # DDMP pipeline封装图像生成全过程：初始化模型，生成随机噪声，逐步去噪
    image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
    image_pipe.to(device)

    # 生成图像
    # 执行image_pipe()会执行以上步骤并返回生成的图像
    # images属性包含图像数据
    images = image_pipe().images

    # 初始化调度器并设置时间步
    scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
    scheduler.set_timesteps(num_inference_steps=40)
    print(scheduler.timesteps)

    # 生成随机图像张量，模拟初始输入的噪声
    # batch_size为4，三通道，长，宽均为256像素的一组图像
    x = torch.randn(4, 3, 256, 256).to(device)

    # 迭代时间步time steps，进行图像生成
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        # 准备模型输入，给带噪图像加上时间步信息
        # 输入图像 x：这是一个包含噪声的图像张量
        # 时间步 t：当前迭代的时间步，用于确定噪声的程度
        # scale_model_input 根据时间步 t 对图像 x 进行缩放，使其符合当前去噪阶段的要求
        model_input = scheduler.scale_model_input(x, t)

        # 预测噪声
        with torch.no_grad():  # 禁用梯度计算，节省内存和计算资源
            #  UNet是扩散模型中的去噪网络,接收带噪图像, 预测图像中的噪声
            noise_pred = image_pipe.unet(model_input, t)["sample"]  # 模型预测的噪声, "sample"表示去噪后的图像张量

        # 使用调度器计算更新后的样本应该是什么样子
        # 根据输入(noise_pred, time_step t和带噪声的图像x)，计算并应用去噪操作，生成更新后的图像
        # 通常包括 prev_sample（更新后的图像）和 pred_original_sample（去噪后的原始图像）
        scheduler_output = scheduler.step(noise_pred, t, x)

        # 更新输入图像，prev_sample（更新后的图像）
        x = scheduler_output.prev_sample

        # 绘制输入图像和预测的去噪图像
        if i % 10 == 0 or i == len(scheduler.timesteps) - 1:
            fig, axs = plt.subplots(1, 2, figsize=(12, 5))
            grid = torchvision.utils.make_grid(x, nrow=4).permute(1, 2, 0)
            axs[0].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
            axs[0].set_title(f"Current x (step {i})")

            pred_x0 = scheduler_output.pred_original_sample  # pred_orig_sample去噪后的原始图像
            grid = torchvision.utils.make_grid(pred_x0, nrow=4).permute(1, 2, 0)
            axs[1].imshow(grid.cpu().clip(-1, 1) * 0.5 + 0.5)
            axs[1].set_title(f"Predicted denoised images (step {i})")
            plt.show()


# 更换新的调度器
def fine_tuning_example_v02():
    # 选择设备
    device = ("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    # 加载预训练模型并设置设备
    # DDMP pipeline封装图像生成全过程：初始化模型，生成随机噪声，逐步去噪
    image_pipe = DDPMPipeline.from_pretrained("google/ddpm-celebahq-256")
    image_pipe.to(device)

    # 生成图像
    # 执行image_pipe()会执行以上步骤并返回生成的图像
    # images属性包含图像数据
    images = image_pipe().images

    # 初始化调度器并设置时间步
    scheduler = DDIMScheduler.from_pretrained("google/ddpm-celebahq-256")
    scheduler.set_timesteps(num_inference_steps=40)
    print(scheduler.timesteps)

    # 替换新的调度器
    image_pipe.scheduler = scheduler
    images = image_pipe(num_inference_steps=40).images
    plt.imshow(images[0])
    plt.show()


# Fine tuning with dataset
def fine_tuning_example_v03():
    dataset_name = "huggan/smithsonian_butterflies_subset"
    dataset = load_dataset(dataset_name, split="train")

    image_size = 1024
    batch_size = 4
    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # 将图像调整为 256x256 大小
            transforms.RandomHorizontalFlip(),  # 随机水平翻转图像，以增加数据的多样性
            transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
            transforms.Normalize([0.5], [0.5]),  # 标准化图像数据，使其均值为 0.5，标准差为 0.5
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    print("Previewing batch:")
    batch = next(iter(train_dataloader))
    grid = torchvision.utils.make_grid(batch["images"], nrow=4)
    plt.imshow(grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5)
    plt.show()


def fine_tuning_example_v04_training_loop(
        image_size=256,
        batch_size=4,
        num_epochs=2,
        lr=1e-5,  # learning rate
        pretrained_model="google/ddpm-celebahq-256",
        dataset_name="huggan/smithsonian_butterflies_subset",
        wandb_project='Fine-tuning-DMs-log',
        fine_tuned_model_name="fine_tuned_model_v01"
):
    # ============================step 1=========================================
    # 选择设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wandb.init(project=wandb_project, config=locals())

    # 加载预训练模型并设置设备
    # DDMP pipeline封装图像生成全过程：初始化模型，生成随机噪声，逐步去噪
    try:
        image_pipe = DDPMPipeline.from_pretrained(pretrained_model, allow_pickle=True)
        image_pipe.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")

    # scheduler = DDIMScheduler.from_pretrained(pretrained_model)
    scheduler = DDIMScheduler.from_config(pretrained_model)
    scheduler.set_timesteps(num_inference_steps=40)

    # image_pipe.scheduler = scheduler  # 替换scheduler

    # =============================step 2========================================
    dataset = load_dataset(dataset_name, split="train")

    preprocess = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),  # 将图像调整为 256x256 大小
            transforms.RandomHorizontalFlip(),  # 随机水平翻转图像，以增加数据的多样性
            transforms.ToTensor(),  # 将图像转换为 PyTorch 张量
            transforms.Normalize([0.5], [0.5]),  # 标准化图像数据，使其均值为 0.5，标准差为 0.5
        ]
    )

    def transform(examples):
        images = [preprocess(image.convert("RGB")) for image in examples["image"]]
        return {"images": images}

    dataset.set_transform(transform)
    train_dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # =============================step 3========================================
    grad_accumulation_steps = 2  # @param
    optimizer = torch.optim.AdamW(image_pipe.unet.parameters(), lr=lr)
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    losses = []

    # 确认所有张量在同一设备上
    def check_tensor(tensor, name):
        assert tensor.device == device, f"{name} on {tensor.device}, expected {device}"
        print(f"{name} shape: {tensor.shape}, device: {tensor.device}")

    for epoch in range(num_epochs):
        for step, batch in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):
            clean_images = batch["images"].to(device)

            # Sample noise to add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bs = clean_images.shape[0]

            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                image_pipe.scheduler.config.num_train_timesteps,
                (bs,),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = image_pipe.scheduler.add_noise(clean_images, noise, timesteps)

            # Get the model prediction for the noise
            noise_pred = image_pipe.unet(noisy_images, timesteps, return_dict=False)[0]

            # print(f"Timesteps shape: {timesteps.shape}, device: {timesteps.device}")
            # print(f"Noisy images shape: {noisy_images.shape}, device: {noisy_images.device}")

            # 检查输入张量的设备和形状
            assert noise_pred.device.type == device.split(':')[
                0], f"Noise prediction is on {noise_pred.device}, expected {device}"
            assert noise.device == clean_images.device, f"Noise is on {noise.device}, expected {clean_images.device}"
            assert noise_pred.shape == noise.shape, f"Noise prediction shape {noise_pred.shape} does not match noise shape {noise.shape}"

            # Compare the prediction with the actual noise:
            loss = F.mse_loss(
                noise_pred, noise
            )  # NB - trying to predict noise (eps) not (noisy_ims-clean_ims) or just (clean_ims)

            # 记录损失
            wandb.log({'loss': loss.item()})

            # print(f"Loss: {loss.item()}")
            # print(f"Clean images shape: {clean_images.shape}, device: {clean_images.device}")
            # print(f"Noise shape: {noise.shape}, device: {noise.device}")
            # print(f"Noisy images shape: {noisy_images.shape}, device: {noisy_images.device}")
            # print(f"Noise prediction shape: {noise_pred.shape}, device: {noise_pred.device}")
            # print(f"Loss gradient device: {loss.grad_fn}")

            # Store for later plotting
            losses.append(loss.item())

            # Update the model parameters with the optimizer based on this loss
            loss.backward(loss)

            # Gradient accumulation:
            if (step + 1) % grad_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        print(f"Epoch {epoch} average loss: {sum(losses[-len(train_dataloader):]) / len(train_dataloader)}")

    # 保存pipe
    image_pipe.save_pretrained(fine_tuned_model_name)

    plt.plot(losses)
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.show()

    wandb.finish()


def try_finetuned_model():
    device = "cuda"
    # 加载微调后的DDPM模型
    model_path = "fine_tuned_model_v01"  # 这里填入你的微调模型的路径

    pipeline = DDPMPipeline.from_pretrained(model_path)
    pipeline.to(device)  # 如果有GPU，使用CUDA

    scheduler = DDIMScheduler.from_pretrained(f"{model_path}/scheduler")
    scheduler.set_timesteps(num_inference_steps=40)

    x = torch.randn(8, 3, 256, 256).to(device)
    for i, t in tqdm(enumerate(scheduler.timesteps)):
        model_input = scheduler.scale_model_input(x, t)
        with torch.no_grad():
            noise_pred = pipeline.unet(model_input, t)["sample"]
        x = scheduler.step(noise_pred, t, x).prev_sample

    save_dir = "finetuned_DMs_generated_images"
    os.makedirs(save_dir, exist_ok=True)

    # 保存生成的图像
    for i in range(x.size(0)):
        img = x[i].permute(1, 2, 0).cpu().numpy()
        img = (img * 255).clip(0, 255).astype("uint8")
        pil_img = Image.fromarray(img)
        pil_img.save(os.path.join(save_dir, f"finetuned_generated_image_{i}.png"))

    fig, axs = plt.subplots(1, 2, figsize=(12, 5))
    grid = torchvision.utils.make_grid(x, nrow=4)
    plt.imshow(grid.permute(1, 2, 0).cpu().clip(-1, 1) * 0.5 + 0.5)
    plt.show()

    # # 生成图像
    # num_images = 1  # 要生成的图像数量
    # generated_images = pipeline(num_images)["sample"]
    #
    # # 保存生成的图像
    # for i, img in enumerate(generated_images):
    #     img = img.permute(1, 2, 0)  # 将图像通道顺序转换为 (H, W, C)
    #     img = (img * 255).clamp(0, 255).numpy().astype("uint8")  # 将图像转换为uint8类型
    #     pil_img = Image.fromarray(img)  # 将numpy数组转换为PIL图像
    #     pil_img.save(f"finetuned_generated_image_{i}.png")  # 保存图像


def main():
    # fine_tuning_example_v03()
    # fine_tuning_example_v04_training_loop()
    try_finetuned_model()


if __name__ == '__main__':
    main()
