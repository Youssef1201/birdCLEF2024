# Afficher la taille du dataset
print(f"There are {len(dataset)} samples in the dataset.")
signal, label = dataset[0]
print(f"Signal shape: {signal.shape}, Label: {label}")
# Instanciation du réseau de neuronnes avec le bon nombre de classes d'oiseaux
num_classes = 264  
model_cnn = CNNNetwork(num_classes,device)
summary(cnn.cuda(), (1, 64, 44))
​
# Définir la fonction de perte et l'optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
​
total_batches = len(dataloader)
​
num_epochs = 10
for epoch in range(num_epochs):
    print(f"Entrainement sur {total_batches} batch de donnée pour l'époque {epoch+1}/{num_epochs}")
    for batch_idx, (inputs, labels) in enumerate(dataloader):
        outputs = model_cnn(inputs)
        labels = labels.reshape(-1)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    # Afficher le progrès de l'entraînement
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")
​
​
print("Entraînement terminé.")
There are 24459 samples in the dataset.
Signal shape: torch.Size([1, 64, 44]), Label: tensor([0], device='cuda:0')
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 66, 46]             160
              ReLU-2           [-1, 16, 66, 46]               0
         MaxPool2d-3           [-1, 16, 33, 23]               0
            Conv2d-4           [-1, 32, 35, 25]           4,640
              ReLU-5           [-1, 32, 35, 25]               0
         MaxPool2d-6           [-1, 32, 17, 12]               0
            Conv2d-7           [-1, 64, 19, 14]          18,496
              ReLU-8           [-1, 64, 19, 14]               0
         MaxPool2d-9             [-1, 64, 9, 7]               0
           Conv2d-10           [-1, 128, 11, 9]          73,856
             ReLU-11           [-1, 128, 11, 9]               0
        MaxPool2d-12            [-1, 128, 5, 4]               0
          Flatten-13                 [-1, 2560]               0
           Linear-14                  [-1, 264]         676,104
          Softmax-15                  [-1, 264]               0
================================================================
Total params: 773,256
Trainable params: 773,256
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.84
Params size (MB): 2.95
Estimated Total Size (MB): 4.80
----------------------------------------------------------------
Entrainement sur 765 batch de donnée pour l'époque 1/10
Epoch 1/10, Loss: 5.576229572296143
Entrainement sur 765 batch de donnée pour l'époque 2/10
Epoch 2/10, Loss: 5.57710075378418
Entrainement sur 765 batch de donnée pour l'époque 3/10
Epoch 3/10, Loss: 5.575735569000244
Entrainement sur 765 batch de donnée pour l'époque 4/10
Epoch 4/10, Loss: 5.576566696166992
Entrainement sur 765 batch de donnée pour l'époque 5/10
Epoch 5/10, Loss: 5.5767717361450195
Entrainement sur 765 batch de donnée pour l'époque 6/10
Epoch 6/10, Loss: 5.578217506408691
Entrainement sur 765 batch de donnée pour l'époque 7/10
Epoch 7/10, Loss: 5.575928211212158
Entrainement sur 765 batch de donnée pour l'époque 8/10
Epoch 8/10, Loss: 5.575989246368408
Entrainement sur 765 batch de donnée pour l'époque 9/10
Epoch 9/10, Loss: 5.576718330383301
Entrainement sur 765 batch de donnée pour l'époque 10/10
Epoch 10/10, Loss: 5.577319622039795
Entraînement terminé.
