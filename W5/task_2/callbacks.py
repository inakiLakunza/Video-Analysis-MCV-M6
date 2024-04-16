
import torch
    

class SaveBestModel:
    def __init__(self, save_path, best_val_loss=1e10):
        self.best_val_loss = best_val_loss
        self.save_path = save_path
        self.best_params = {}

    def check(self, val_loss, model, optimizer, epoch):
        self.best_params = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': val_loss,
        }
        self.best_val_loss = val_loss
    
    def save(self, val_loss, model, optimizer, epoch):

        # CHECK IF THE MODEL FROM THE LAST EPOCH IS BEST
        if len(self.best_params == 0):
            self.best_params = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': val_loss,
            }

        torch.save(self.best_params, self.save_path)

    

class EarlyStopper:
    def __init__(self, save_path, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')
        self.stop = False
        self.model_saver = SaveBestModel(save_path)

    def early_stop(self, validation_loss, model, optimizer, epoch):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            # Guardar parameters
            self.model_saver.check(validation_loss, model, optimizer, epoch)

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                self.stop = True


    def get_stop(self):
        return self.stop