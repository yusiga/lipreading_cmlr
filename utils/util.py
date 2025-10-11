import editdistance
from loguru import logger
import torch

# from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

# from data.dataset_cmlr_base import CmlrSet


class Util:
    @staticmethod
    def count_parameters(model):
        """
        计算 model 可训练的参数
        """
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    @staticmethod
    def er(predict, truth):
        element_pairs = [(p[0].split(" "), p[1].split(" ")) for p in zip(predict, truth)]
        er = [1.0 * editdistance.eval(p[0], p[1]) / len(p[1]) for p in element_pairs]
        return er

    @staticmethod
    def log_args(args):
        """
        通过loguru 记录 args 各项参数
        """
        logger.add("{}/args.log".format(args.save_root), format="{message}")
        for arg in vars(args):
            logger.warning("{}: {}".format(arg, getattr(args, arg)))

    # @staticmethod
    # def init_loader(dataset, mode, args):
    #     """
    #     通过 dataset 初始化 dataloader
    #     """
    #     shuffle = False
    #     if mode == "train":
    #         shuffle = True

    #     data_loader = DataLoader(
    #         dataset=dataset,
    #         num_workers=args.num_workers,
    #         pin_memory=args.pin_memory,
    #         batch_size=args.batch_size,
    #         shuffle=shuffle,
    #         collate_fn=CmlrSet.collate_fn,
    #     )
    #     logger.debug("{m} 数据集的样本数: {l}".format(m=mode, l=len(data_loader.dataset)))
    #     return data_loader

    @staticmethod
    def save_state(model, optimizer, scheduler_reduce, scheduler_warmup, epoch, ckpt_file):
        """
        保存 model, optimizer, scheduler, epoch, loss 状态
        """
        states = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_reduce_state_dict": scheduler_reduce.state_dict(),
            "scheduler_warmup_state_dict": scheduler_warmup.state_dict(),
            "epoch": epoch,
        }
        torch.save(obj=states, f=ckpt_file)

    @staticmethod
    def load_state(model, optimizer, scheduler_reduce, scheduler_warmup, ckpt_file, device):
        """
        从 checkpoint 中加载 model, optimizer, scheduler, epoch, loss
        """
        checkpoint = torch.load(ckpt_file)

        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler_reduce.load_state_dict(checkpoint["scheduler_reduce_state_dict"])
        scheduler_warmup.load_state_dict(checkpoint["scheduler_warmup_state_dict"])
        epoch_start = checkpoint["epoch"] + 1

        # dvc = torch.device(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.to(device)

        return model, optimizer, scheduler_reduce, scheduler_warmup, epoch_start

    @staticmethod
    def load_model_state(model, ckpt_file):
        """
        从 checkpoint 中加载 model
        """
        # checkpoint = torch.load(ckpt_file)
        checkpoint = torch.load(ckpt_file,map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model'])
        return model

    @staticmethod
    def unique(tensor, padding_value=0):
        # 3D --> 2D
        B = tensor.shape[0]
        tensor = tensor.view(B, -1)
        # unipue
        tensor_list = [torch.unique(arr) for arr in tensor]
        # pad 0
        tensor = pad_sequence(tensor_list, batch_first=True, padding_value=padding_value)
        return tensor
