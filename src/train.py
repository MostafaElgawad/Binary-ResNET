import torch
import torch.optim as optim
import os
from os.path import join as ospj
from torch.utils.tensorboard import SummaryWriter
from .model import *

class Trainer:
    def __init__(self, model, epochs, train_loader, val_loader, patience:int=5):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model = model.to(self.device)
        self.epochs = epochs
        self.patience = patience
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        # Binary Cross Entropy Loss since two classes
        self.criterion = nn.BCEWithLogitsLoss()


        self.start_epoch = 1
        self.best_epoch = 1
        self.current_epoch = 1

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.writer = SummaryWriter(ospj('logs'))

        self.checkpoints_path = ospj(
            "checkpoints","Best_Model"
        )
        if not os.path.exists(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)

    def _train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        for images, labels in self.train_loader:
            images, labels = images.to(self.device), labels.to(self.device)
            labels = labels.float()

            self.optimizer.zero_grad()
            outputs = self.model(images)
            #flatten output to be fed to the BCE loss
            outputs = outputs.view(-1)

            loss = self.criterion(outputs, labels)

            loss.backward()
            self.optimizer.step()

            # Calculate train accuracy
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0) # accumalting number of total samples processed
            total_loss += loss.item()
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total

        # Log metrics to TensorBoard
        self.writer.add_scalar('Loss/train', avg_loss, self.current_epoch)
        self.writer.add_scalar('Accuracy/train', accuracy, self.current_epoch)
        
        return {'loss': avg_loss, 'accuracy': accuracy}


    @torch.no_grad()
    def evaluate(self, loader=None):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        if loader is None:
            assert self.val_loader is not None, 'loader was not given and self._eval_loader not set either!'
            loader = self.val_loader
        for images, labels in loader:
            images, labels = images.to(self.device), labels.to(self.device)
            labels = labels.float()

            outputs = self.model(images)
            outputs = outputs.view(-1)

            loss = self.criterion(outputs, labels)
            
            predictions = (torch.sigmoid(outputs) > 0.5).float()

            all_preds.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            total_loss += loss.item()


        avg_loss = total_loss / len(loader)
        accuracy = correct / total

        # Log metrics to TensorBoard
        self.writer.add_scalar('Loss/val', avg_loss, self.current_epoch)
        self.writer.add_scalar('Accuracy/val', accuracy, self.current_epoch)
        return {'loss': avg_loss, 'accuracy': accuracy}, all_preds, all_labels 
        
    def train(self):
        self.not_improved_count = 0
        best_val_accuracy = float('-inf') 

        for epoch in range(self.start_epoch, self.epochs + 1):
            self.current_epoch = epoch
            # train for one epoch then evaluate 
            train_metrics = self._train_epoch()
            #evalute
            val_metrics, _ , _ = self.evaluate()

            #progress
            print(f'Epoch {epoch}/{self.epochs}:')
            print(f'Train Loss: {train_metrics["loss"]:.4f}, Train Acc: {train_metrics["accuracy"]:.4f}')
            print(f'Val Loss: {val_metrics["loss"]:.4f}, Val Acc: {val_metrics["accuracy"]:.4f}')

            #checking on improvments
            if val_metrics['accuracy'] > best_val_accuracy:
                best_val_accuracy = val_metrics['accuracy']
                self.best_epoch = epoch
                self.not_improved_count = 0

                # Save best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_accuracy': best_val_accuracy,
                },  os.path.join(self.checkpoints_path,'model.pth'))
            else:
                self.not_improved_count += 1

            #early stopping
            if self.not_improved_count >= self.patience:
                print(f'Early stopping triggered after epoch {epoch}')
                break

        self.writer.close()
        print(f'Training completed. Best validation accuracy: {best_val_accuracy:.4f} at epoch {self.best_epoch}')

    def load_model(self, path=None):
        self.model.load_state_dict(torch.load(path))
