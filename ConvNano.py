
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch import nn
from torch import optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.nn.functional as F
from torchvision import transforms as T
from torchvision import models
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import confusion_matrix
import glob
from astropy.visualization import simple_norm
from tqdm import tqdm
from skimage.transform import resize
from BinaryMergerDataset import BinaryMergerDataset, get_transforms
import pickle
from SetRandomSeed import set_random_seeds, GeneratorSeed
from zoobot.pytorch.training.finetune import FinetuneableZoobotClassifier
import argparse


set_random_seeds(626)
g = GeneratorSeed(626)
path = '/n/holystore01/LABS/hernquist_lab/Users/aschechter/NormalizedDataset/'
pad_val = 0
BATCH_SIZE = 128
VALIDATION_BATCH_SIZE = 128
NUM_EPOCHS = 300
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(device)
num_workers = 6
num_classes=2

torch.backends.cudnn.benchmark = True  # ADD THIS
torch.backends.cudnn.allow_tf32 = True 
torch.backends.cuda.matmul.allow_tf32 = True



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--layer_decay', type=float, default=0.5)
    parser.add_argument('--dropout', type = float, default=0.2)
    parser.add_argument('--weight_decay', type = float, default=0.001)
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--trial_number', type=int, default = 999)
    parser.add_argument('--label_smoothing', type=float, default=0.1)
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to a .pth checkpoint to resume training from')
    return parser.parse_args()


def get_accuracy_BCE(pred,original):
    pred = (torch.sigmoid(pred) > 0.5).float() #pred is now 0 or 1
    pred = pred.cpu().detach().numpy()
    original = original.cpu().numpy()

    return torch.mean(pred == original) * 100

def get_accuracy_CE(pred,original):
    return np.mean(pred.cpu().numpy() == original.long().cpu().numpy()) * 100

def plot_confusion_matrix(cm, classes, epoch, learning_rate, modelname): #help from chat GPT
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=classes, yticklabels=classes)
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.title('Confusion Matrix After' + str(epoch) + 'Epochs')
    plt.savefig('ConfusionMatrix_'+ modelname +' Adam' + str(learning_rate) + '.png', dpi = 300)

class EarlyStopping:
    def __init__(self, patience=15, delta=0, max_gap=0.08, gap_patience=10, keep_top_k=3):
        self.patience = patience
        self.delta = delta
        self.max_gap = max_gap
        self.gap_patience = gap_patience
        self.keep_top_k = keep_top_k
        self.best_loss = float('inf')
        self.early_stop = False
        self.counter = 0
        self.gap_counter = 0
        self.best_epoch = -1 # Added this
        self.freeze_until = -1  # epoch below which early stopping is suppressed
        # Store top-k models as a list of (loss, gap, state_dict, epoch) tuples
        self.top_k_models = []

    def __call__(self, val_loss, train_loss, model, epoch, ):
        gap = val_loss - train_loss
        model_info = {
            'val_loss': val_loss,
            'train_loss': train_loss,
            'gap': gap,
            'state_dict': {k: v.clone() for k, v in model.state_dict().items()},
            'epoch': epoch
        }
        self.top_k_models.append(model_info)

        # Sort by validation loss and keep only top-k
        self.top_k_models.sort(key=lambda x: x['val_loss'])
        self.top_k_models = self.top_k_models[:self.keep_top_k]

        # Update best_loss
        old_best = self.best_loss
        self.best_loss = self.top_k_models[0]['val_loss']

        if epoch < self.freeze_until:
            print(f'Warmup period: early stopping suppressed until epoch {self.freeze_until} (currently epoch {epoch})')
            return

        # Check if we improved
        if self.best_loss < old_best - self.delta:
            self.best_epoch = self.top_k_models[0]['epoch']
            self.counter = 0
        else:
            self.counter += 1

        # Stop if no improvement for patience epochs
        if self.counter >= self.patience:
            self.early_stop = True
            print(f'Early stopping triggered at epoch {epoch}: no improvement in val_loss for {self.patience} epochs (best val_loss={self.best_loss:.4f} at epoch {self.best_epoch})')

        if gap > self.max_gap:
            self.gap_counter += 1
            print(f'Overfitting gap {gap:.4f} > {self.max_gap} at epoch {epoch} ({self.gap_counter}/{self.gap_patience})')
            if self.gap_counter >= self.gap_patience:
                self.early_stop = True
                print(f'Gap-based early stopping triggered at epoch {epoch}')
        else:
            self.gap_counter = 0

    def get_best_model(self, rank=0, max_gap=None):
        """
        Get the rank-th best model by val_loss, optionally filtered by gap.
        If max_gap is set, only consider models with gap <= max_gap.
        Falls back to rank-0 if no model passes the gap filter.
        """
        candidates = self.top_k_models
        if max_gap is not None:
            filtered = [m for m in candidates if m['gap'] <= max_gap]
            if filtered:
                candidates = filtered
            else:
                print(f"No models with gap <= {max_gap}; using best by val_loss")
        if rank < len(candidates):
            return candidates[rank]
        return None
    
    def load_best_model(self, model, rank=0, max_gap=None):
        """Load the rank-th best model, optionally filtered by gap"""
        best_model = self.get_best_model(rank, max_gap=max_gap)
        if best_model:
            model.load_state_dict(best_model['state_dict'])
            print(f"Loaded model from epoch {best_model['epoch']} "
                  f"(val_loss: {best_model['val_loss']:.4f}, "
                  f"gap: {best_model['gap']:.4f})")
        else:
            print(f"No model available at rank {rank}")
    
    def save_all_models(self, optimizer, lr_scheduler, modelname, learning_rate):
        """Save all top-k models"""
        for i, model_info in enumerate(self.top_k_models):
            filename = f"{modelname}_{learning_rate}_rank{i}_epoch{model_info['epoch']}.pth"
            torch.save({
                'epoch': model_info['epoch'],
                'model_state_dict': model_info['state_dict'],
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'val_loss': model_info['val_loss'],
                'train_loss': model_info['train_loss'],
                'gap': model_info['gap'],
                'rank': i
            }, filename)
            print(f"Saved rank {i} model: {filename} "
                  f"(epoch {model_info['epoch']}, val_loss: {model_info['val_loss']:.4f}, "
                  f"gap: {model_info['gap']:.4f})")
            
            
                     
def draw_curve(current_epoch, modelloss, modelacc, x_epoch, fig, ax0, ax1, learning_rate, batch_size, modelname):
    """
    Updates and saves training/validation loss and accuracy curves.
    """
    ax0.clear()
    ax1.clear()
    
    x_epoch.append(current_epoch)

    ax0.plot(x_epoch, modelloss['train'], label='train', color='rebeccapurple')
    ax0.plot(x_epoch, modelloss['validation'], label='val', color='darkorange')
    ax1.plot(x_epoch, modelacc['train'], label='train', color='rebeccapurple')
    ax1.plot(x_epoch, modelacc['validation'], label='val', color='darkorange')

    ax1.axhline(y=65, color='grey', linestyle='--')
    ax1.axhline(y=75, color='grey', linestyle='--')

    # y-axis scaling
    if np.max(modelloss['validation']) > 5:
        ax0.set_ylim(0, 5)
    else:
        ax0.set_ylim(np.min(modelloss['validation']) - 0.25, np.max(modelloss['validation']) + 0.25)
    ax1.set_ylim(30, 100)

    # add legends only once
    if current_epoch == 0:
        ax0.legend()
        ax1.legend()

    plt.tight_layout()
    fig.savefig(f'metrics_{modelname}_Adam{learning_rate}_batch{batch_size}.png', dpi=300)
    
    
def main():
    args = parse_args()
    learning_rate = args.lr
    lr_decay = args.layer_decay
    dropout = args.dropout
    weight_decay = args.weight_decay
    BATCH_SIZE = args.batch_size
    hidden_size = args.hidden_size
    trial_number = args.trial_number
    label_smoothing = args.label_smoothing
    checkpoint_path = args.checkpoint
    
    train_mergers_dataset_augment = BinaryMergerDataset(path, 'train', mergers = True, transform = get_transforms(aug=True), codetest=False)
    train_nonmergers_dataset_augment = BinaryMergerDataset(path, 'train', mergers = False, transform = get_transforms(aug=True), codetest=False)

    train_mergers_dataset_augment2 = BinaryMergerDataset(path, 'train', mergers = True, transform = get_transforms(aug=True), codetest=False)
    train_nonmergers_dataset_augment2 = BinaryMergerDataset(path, 'train', mergers = False, transform = get_transforms(aug=True), codetest=False)

    train_mergers_dataset_augment3 = BinaryMergerDataset(path, 'train', mergers = True, transform = get_transforms(aug=True), codetest=False)
    train_nonmergers_dataset_augment3 = BinaryMergerDataset(path, 'train', mergers = False, transform = get_transforms(aug=True), codetest=False)

    train_mergers_dataset_orig = BinaryMergerDataset(path, 'train', mergers = True, transform = get_transforms(aug=False), codetest=False)
    train_nonmergers_dataset_orig = BinaryMergerDataset(path, 'train', mergers = False, transform = get_transforms(aug=False), codetest=False)

    train_dataset_full = torch.utils.data.ConcatDataset([train_mergers_dataset_augment, train_nonmergers_dataset_augment, train_mergers_dataset_augment2, train_nonmergers_dataset_augment2,  train_mergers_dataset_augment3, train_nonmergers_dataset_augment3, train_mergers_dataset_orig, train_nonmergers_dataset_orig])
    n_train_mergers = 4 * len(train_mergers_dataset_orig) 
    n_train_nonmergers = 4 * len(train_nonmergers_dataset_orig) 
    print(f'Training set: {n_train_mergers} mergers, {n_train_nonmergers} nonmergers, {len(train_dataset_full)} total')
    train_dataloader = DataLoader(train_dataset_full, shuffle = True, num_workers = num_workers, batch_size=BATCH_SIZE, generator=g,pin_memory=True, persistent_workers=True
                                  )


    validation_mergers_dataset_orig = BinaryMergerDataset(path, 'validation', mergers = True, transform = get_transforms(aug=False), codetest=False)
    validation_nonmergers_dataset_orig = BinaryMergerDataset(path, 'validation', mergers = False, transform = get_transforms(aug=False), codetest=False)

    validation_dataset_full = torch.utils.data.ConcatDataset([validation_mergers_dataset_orig, validation_nonmergers_dataset_orig])
    indices = np.random.permutation(len(validation_dataset_full))
    shuffled_validation_dataset = Subset(validation_dataset_full, indices)
    validation_dataloader = DataLoader(validation_dataset_full, shuffle = True, num_workers = num_workers, batch_size=BATCH_SIZE, generator = g,
                                       pin_memory=True, persistent_workers=False)#num workers used to be 4



    total_size_train = len(train_dataset_full)
    total_size_validation = len(validation_dataset_full)

    print(total_size_train)

    number_of_iterations = total_size_train//BATCH_SIZE #// is floor
    print('number_of_iterations', number_of_iterations)

    
    modelname = 'nano_' + str(trial_number)

    model = FinetuneableZoobotClassifier(name='hf_hub:mwalmsley/zoobot-encoder-convnext_nano', learning_rate=learning_rate,  # use a low learning rate
        layer_decay=lr_decay,  # reduce the learning rate from lr to lr^0.5 for each block deeper in the network
     
        num_classes=2
    )
    model.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(640, hidden_size), 
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_classes)
        )

    
    
    model = model.to(device)


    modelloss = {} #loss history
    modelloss['train'] = []
    modelloss['validation'] = []
    modelacc = {} #used later for accuracy
    modelacc['train'] = []
    modelacc['validation'] = []
    x_epoch = []

    columndata = {}
    columndata['accuracy_column'] = []
    columndata['output_column'] = []
    columndata['label_column'] = []




    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay= weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=30,              # Restart every 30 epochs
        T_mult=1,            # Constant cycle length
        eta_min=learning_rate*0.1  # Minimum LR
    )

    
    scaler = torch.cuda.amp.GradScaler()
    start_epoch = 0
    if checkpoint_path is not None:
        print(f'Loading checkpoint: {checkpoint_path}')
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        for pg in optimizer.param_groups:
            pg['lr'] = learning_rate
            pg['weight_decay'] = weight_decay
        if 'lr_scheduler_state_dict' in checkpoint:
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])
        else:
            print('Warning: checkpoint has no lr_scheduler_state_dict; scheduler starts fresh')
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        print(f'Resuming from epoch {start_epoch}')

    fig = plt.figure(figsize=(12, 6))
    ax0 = fig.add_subplot(121, title="Loss")
    ax1 = fig.add_subplot(122, title="Accuracy")
    plt.suptitle(f'lr={learning_rate} | batch={BATCH_SIZE}')


    best_val_loss = float(np.inf)  # Initialize to a very large value
    best_model_state = None
    best_epoch = -1


    earlystop = EarlyStopping(patience=25, delta=0.0001, max_gap=0.05, gap_patience=10, keep_top_k=3)
    for epoch in range(start_epoch, NUM_EPOCHS):
        t_epoch_loss = 0.0
        v_epoch_loss = 0.0
        t_accuracy = []
        v_accuracy = []
        t_counter = 0
        v_counter = 0
        model.train(True)
        for images, labels, names in tqdm(iter(train_dataloader)):
            bs = images.shape[0]         #batch size
            images = images.to(device=device, dtype=torch.float32)
            labels = labels.to(device=device, dtype=torch.float32)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = F.cross_entropy(outputs, labels.long(), label_smoothing=label_smoothing)

            pred = torch.argmax(outputs, dim=1).to(dtype=torch.float32)
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            t_epoch_loss += loss.item() * bs #make the loss a number
            t_accuracy.append(get_accuracy_CE(pred, labels))
            t_counter +=1 
  
        lr_scheduler.step()
 
        modelacc['train'].append(np.mean(np.array(t_accuracy)))

   
        modelloss['train'].append(t_epoch_loss/total_size_train)

        print('TRAINING LOSS: ', t_epoch_loss)

        model.eval()
        all_labels = []
        all_preds = []
        all_features = [] 

        with torch.no_grad():
            for images, labels, names in tqdm(iter(validation_dataloader)):
       
                bs = images.shape[0] 
             
                images = torch.tensor(images, dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.float32).to(device)
               
                outputs = model(images)
        
                pred = torch.argmax(outputs, dim=1) 
                all_labels.extend(labels.cpu().numpy())
            
                all_preds.extend(outputs.detach().cpu().numpy())
                all_features.extend(pred.detach().cpu().numpy())
    
                labels = labels.to(device=device, dtype=torch.float32)
                loss = F.cross_entropy(outputs, labels.long(), label_smoothing=label_smoothing)
                v_epoch_loss += loss.item() * bs
                v_accuracy.append(get_accuracy_CE(pred, labels))
                v_counter += 1
              
        modelacc['validation'].append(np.mean(np.array(v_accuracy)))

        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        all_features = np.array(all_features)
        all_preds = np.rint(all_preds)

        modelloss['validation'].append(v_epoch_loss/total_size_validation)

        if v_epoch_loss/total_size_validation < best_val_loss:
            best_val_loss = v_epoch_loss/total_size_validation
            best_model_state = model.state_dict()
            best_epoch = epoch

        #want to draw curve and then stop the loop
        draw_curve(epoch, modelloss, modelacc, x_epoch, fig, ax0, ax1, learning_rate, BATCH_SIZE, modelname)

        if epoch == 100:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'train_loss': t_epoch_loss / total_size_train,
                'val_loss': v_epoch_loss / total_size_validation,
            }, modelname + '_trial' + str(trial_number) + '_lr' + str(learning_rate) + '_epoch100.pth')
            print(f'Saved full checkpoint at epoch 100')
            
        if epoch == 120:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': t_epoch_loss / total_size_train,
                'val_loss': v_epoch_loss / total_size_validation,
            }, modelname + '_trial' + str(trial_number) + '_lr' + str(learning_rate) + '_epoch120.pth')
            print(f'Saved full checkpoint at epoch 120')

        if epoch == 150:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'train_loss': t_epoch_loss / total_size_train,
                'val_loss': v_epoch_loss / total_size_validation,
            }, modelname + '_trial' + str(trial_number) + '_lr' + str(learning_rate) + '_epoch150.pth')
            print(f'Saved full checkpoint at epoch 150')



        earlystop(val_loss=v_epoch_loss/total_size_validation, train_loss= t_epoch_loss/total_size_train, model=model, epoch=epoch)
        if earlystop.early_stop:
            break
        
    earlystop.load_best_model(model, max_gap=earlystop.max_gap)
    ##save for reloading and plots
    loss_file = open(modelname+ '_trial' + str(trial_number) + '_'+str(learning_rate)+'_loss', 'wb') 
    pickle.dump(modelloss, loss_file) 
    loss_file.close() 

    acc_file = open(modelname+ '_trial' + str(trial_number) + '_'+str(learning_rate)+'_acc', 'wb') 
    pickle.dump(modelacc, acc_file) 
    acc_file.close() 
    
    earlystop.save_all_models(optimizer, lr_scheduler, modelname, learning_rate)
    
    torch.save({
    'epoch': earlystop.best_epoch,  # This will be the last epoch before early stopping
    'model_state_dict': model.state_dict(),  # Now contains the best model
    'optimizer_state_dict': optimizer.state_dict(),
    'scaler_state_dict': scaler.state_dict(),
    'loss': earlystop.best_loss,
    }, modelname + '_trial' + str(trial_number) +  'lr' +str(learning_rate) + '.pth')
    
     

if __name__ == "__main__":
    main()