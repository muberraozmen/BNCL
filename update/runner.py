import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['ModelRunner']


class ModelRunner(object):
    def __init__(self, model, loss, device='cpu', **kwargs):
        self.model = model.to(device)
        self.loss = loss
        self.device = device
        self.optimizer, self.scheduler = self.set_optimizer(**kwargs)

    def train_epoch(self, train_loader, annotated_loader=None):
        epoch_loss = 0
        num_samples = 0
        for step, (entailment, contradiction) in enumerate(train_loader):
            self.model.train()
            self.optimizer.zero_grad()
            entailment_updated, contradiction_updated = self.model(entailment.to(self.device), contradiction.to(self.device))
            batch_loss = self.loss.calculate(entailment_updated.to(self.device), contradiction_updated.to(self.device), device = self.device)
            batch_loss.backward()
            self.optimizer.step()
            epoch_loss += batch_loss.item()
        num_samples += len(train_loader)
        if annotated_loader is not None:
            for step, (entailment, contradiction, targets) in enumerate(annotated_loader):
                self.model.train()
                self.optimizer.zero_grad()
                entailment_updated, contradiction_updated = self.model(entailment.to(self.device), contradiction.to(self.device))
                annotated_loss = self.loss.calculate(entailment_updated, contradiction_updated, y_true=targets)
                annotated_loss.backward()
                self.optimizer.step()
                epoch_loss += annotated_loss.item()
            num_samples += len(annotated_loader)
        if self.scheduler:
            self.scheduler.step()
        return epoch_loss/num_samples

    def test(self, entailments, contradictions):
        self.model.eval()
        with torch.no_grad():
            entailments, contradictions = self.model(entailments.to(self.device), contradictions.to(self.device))
            predictions = self.model.predict(entailments, contradictions)
        return predictions

    def set_optimizer(self, lr=1e-3, betas=(0.8, 0.9), weight_decay=0,
                      step_size=10, gamma=0.1, **kwargs):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma, verbose=False)
        return optimizer, scheduler

    def save_model(self, model_dir):
        torch.save(self.model.state_dict(), model_dir)

    def load_model(self, model_dir):
        self.model.load_state_dict(torch.load(model_dir))


