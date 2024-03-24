from ..logger import Logger
from .BaseTrainer import BaseTrainer


class ClassificationTrainer(BaseTrainer):
    def __init__(self, *args, **kwargs):
        super(ClassificationTrainer, self).__init__(*args, **kwargs)

    def training_step(self, batch):
        image, target = batch
        image, target = image.to(self.device), target.to(self.device)

        output = self.model(image)
        loss, loss_dict = self.criterion(output, target)

        return loss, loss_dict

