import sys
import psutil
import os
import psutil
import time
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from fvcore.nn import FlopCountAnalysis, parameter_count
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, classification_report

class Logger(object):
    def __init__(self, filename="resultados_cnn.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

sys.stdout = Logger("resultados_cnn.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Dispositivo utilizado: ", device)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

transform_test = transforms.Compose(
    [transforms.Resize(size=(164,164)),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)

transform_train = transforms.Compose(
    [transforms.Resize(size = (164,164)),
     transforms.RandomRotation(degrees=10),
     transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.1),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
)

class TransformedSubset(Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.subset)

full_dataset = datasets.ImageFolder(root='dataset', transform=None)
print("\nInformações sobre o Dataset completo: \n\n", full_dataset)
print("\nRótulos: ", full_dataset.class_to_idx)

def cnn_model():
    model = nn.Sequential(
        nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(),

        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(),

        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2, padding=0),

        nn.Flatten(),

        nn.Linear(256 * 41 * 41, 256),
        nn.Dropout(p=0.3),
        nn.ReLU(),

        nn.Linear(256, 128),
        nn.Dropout(p=0.3),
        nn.ReLU(),

        nn.Linear(128, 64),
        nn.Dropout(p=0.3),
        nn.ReLU(),

        nn.Linear(64, 32),
        nn.Dropout(p=0.3),
        nn.ReLU(),

        nn.Linear(32, 3)
    )
    return model

def reset_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        m.reset_parameters()

num_epoch = 200
learning_rate = 0.001

k_folds = 10
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)

classes_names = ['Normal', 'Pneumonia', 'Tuberculosis']

results_acc = []
results_precision_class = {0: [], 1: [], 2: []}
results_recall_class = {0: [], 1: [], 2: []}
results_f1_class = {0: [], 1: [], 2: []}

results_train_acc = []
results_precision_macro = []
results_recall_macro = []
results_f1_macro = []

print("\nAnalisando o custo computacional do modelo...")
temp_model = cnn_model().to(device)

dummy_input = torch.randn(1, 3, 164, 164).to(device)

flops_analysis = FlopCountAnalysis(temp_model, (dummy_input,))

total_flops = flops_analysis.total() * 2 
gflops = total_flops / 1e9

total_params = sum(p.numel() for p in temp_model.parameters())

print(f"==================================================")
print(f"[PERFIL DO MODELO - FVCORE]")
print(f"Total de Parâmetros: {total_params:,}")
print(f"Custo Computacional: {gflops:.4f} GFLOPS (por inferência/imagem)")
print(f"==================================================\n")

del temp_model, dummy_input
if torch.cuda.is_available():
    torch.cuda.empty_cache()

training_start_time = time.time()

for fold, (train_idx, test_idx) in enumerate(kf.split(full_dataset)):
    print(f'\nFold {fold+1}/{k_folds}')
    print(f'\nQuantidade de dados (Treinamento): {len(train_idx)}')
    print(f'Quantidade de dados (Teste): {len(test_idx)}')

    model = cnn_model().to(device)
    model.apply(reset_weights)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    train_subset = Subset(full_dataset, train_idx)
    test_subset = Subset(full_dataset, test_idx)

    train_data = TransformedSubset(train_subset, transform=transform_train)
    test_data = TransformedSubset(test_subset, transform=transform_test)

    trainloader = DataLoader(train_data, batch_size=16, shuffle=True)
    testloader = DataLoader(test_data, batch_size=16, shuffle=False)

    all_targets = []
    for img, rtl, in trainloader:
        all_targets.extend(rtl.tolist())

    n, p, t = [], [], []
    for i in range(len(train_subset)):
        if all_targets[i] == 0:
            n.append(all_targets[i])
        elif all_targets[i] == 1:
            p.append(all_targets[i])
        elif all_targets[i] == 2:
            t.append(all_targets[i])

    print("\n!!!Distribuição dos dados de treinamento!!!\n")
    print(f'Normal: {len(n)}')
    print(f'Pneumonia: {len(p)}')
    print(f'Tuberculose: {len(t)}\n')

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)

    train_losses = []
    train_acc = []

    for epoch in range(num_epoch):
        model.train()
        running_train_loss=0.0
        total_samples = 0
        all_preds_train = []
        all_labels_train = []

        for inputs_train, labels_train in trainloader:
            inputs_train = inputs_train.to(device)
            labels_train = labels_train.to(device)
    
            optimizer.zero_grad()
            outputs_train = model(inputs_train)
            loss = loss_fn(outputs_train, labels_train)
            loss.backward()
            optimizer.step()

            batch_size = inputs_train.size(0)
            running_train_loss += loss.item() * batch_size
            total_samples += batch_size

            _, predicted_train = torch.max(outputs_train, 1)
            all_preds_train.extend(predicted_train.cpu().numpy())
            all_labels_train.extend(labels_train.cpu().numpy())

        train_loss = running_train_loss / total_samples
        train_losses.append(train_loss)

        acc_train = accuracy_score(all_labels_train, all_preds_train)
        train_acc.append(acc_train)

        print(f"Época {epoch + 1}/{num_epoch} - Perda no treinamento: {train_loss:.6f} - Acc: {100 * acc_train:.2f}%")

    results_train_acc.append(train_acc[-1])

    epochs = range(1, num_epoch + 1)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'bo-')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_acc, 'ro-')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.suptitle("Trainning", fontsize = 20)
    plt.savefig(f'train_loss_acc_fold-{fold+1}.png', bbox_inches='tight')

    model.eval()
    with torch.no_grad():
        all_preds_test = []
        all_labels_test = []

        for images_test, labels_test in testloader:
            images_test = images_test.to(device)
            labels_test = labels_test.to(device)       
            outputs_test = model(images_test)
            _, predicted_test = torch.max(outputs_test, 1)
            all_preds_test.extend(predicted_test.cpu().numpy())
            all_labels_test.extend(labels_test.cpu().numpy())

        acc_test = accuracy_score(all_labels_test, all_preds_test)
        results_acc.append(acc_test)

        precision_test = precision_score(all_labels_test, all_preds_test, average=None, zero_division=0)
        recall_test = recall_score(all_labels_test, all_preds_test, average=None, zero_division=0)
        f1_test = f1_score(all_labels_test, all_preds_test, average=None, zero_division=0)

        results_precision_macro.append(sum(precision_test) / 3)
        results_recall_macro.append(sum(recall_test) / 3)
        results_f1_macro.append(sum(f1_test) / 3)
    
    print(f'\n--- Resultados do Fold {fold+1} ---')
    print(f'Acurácia Global: {100 * acc_test:.2f}%\n')

    print(classification_report(all_labels_test, all_preds_test, target_names=classes_names, zero_division=0))

    for i in range(3):
        results_precision_class[i].append(precision_test[i])
        results_recall_class[i].append(recall_test[i])
        results_f1_class[i].append(f1_test[i])

    cm = confusion_matrix(all_labels_test, all_preds_test)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.xlabel('Expected Label')
    plt.ylabel('True Label')
    plt.savefig(f'test_matrizconfusao_fold-{fold+1}.png', bbox_inches='tight')
    plt.close()

    if torch.cuda.is_available():
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
        print(f"Pico Máximo de Memória (VRAM GPU): {peak_mem_mb:.2f} MB")
    else:
        process = psutil.Process(os.getpid())
        mem_mb = process.memory_info().rss / (1024 ** 2)
        print(f"Uso de Memória (RAM Sistema): {mem_mb:.2f} MB")

    print(f'!!!Teste do Fold {fold+1} finalizado!!!')

training_time = time.time() - training_start_time
print(f'\n========================================================')
print(f"\nTempo total de treinamento (10 FOLDS): {training_time:.2f} segundos")
print(f'\n========================================================')

print(f'\n--- Médias finais após {k_folds} FOLDS ---')
print(f'Acurácia Global Média: {100 * (sum(results_acc) / k_folds):.2f}%\n')

print("Desempenho médio por classe:")
print("-" * 50)
print(f"{'Classe':<15} | {'Precisão':<10} | {'Recall':<10} | {'F1-Score':<10}")
print("-" * 50)

for i, class_name in enumerate(classes_names):
    avg_prec = sum(results_precision_class[i]) / k_folds
    avg_rec = sum(results_recall_class[i]) / k_folds
    avg_f1 = sum(results_f1_class[i]) / k_folds

    print(f'{class_name:<15} | {100 * avg_prec:.2f}% | {100 * avg_rec:.2f}% | {100 * avg_f1:.2f}%')

print("-" * 50)

print("\nGerando Boxplot comparativo...")

dados_boxplot = [
    [acc * 100 for acc in results_train_acc],
    [acc * 100 for acc in results_acc],
    [prec * 100 for prec in results_precision_macro],
    [rec * 100 for rec in results_recall_macro],
    [f1 * 100 for f1 in results_f1_macro]
]

labels = ['Train\n(Acc)', 'Test\n(Acc)', 'Test\n(Precision)', 'Test\n(Recall)', 'Test\n(F1-Score)']

plt.figure(figsize=(10, 6))

box = plt.boxplot(dados_boxplot, labels=labels, patch_artist=True,
                  boxprops=dict(facecolor='lightblue', color='blue'),
                  medianprops=dict(color='red', linewidth=2),
                  whiskerprops=dict(color='blue'),
                  capprops=dict(color='blue'))

plt.ylim(80, 100)
plt.yticks(np.arange(80, 101, 10))

plt.ylabel('Performance (%)', fontsize=12)
plt.title('Distribution of Metrics Across the 10 Folds', fontsize=14)

plt.grid(axis='y', linestyle='--', alpha=0.7)

plt.savefig('cnn_boxplot_metricas_folds.png', bbox_inches='tight')
plt.show()
