################################# TASK 3: Remove Overfitting #######################################
nn.Dropout(0.5)
nn.BatchNorm2d()
transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(32, padding=4),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),
                         (0.5,0.5,0.5))
#################### COMPARED THE MODEL ########################################################
   plt.plot(train_acc, label="Train Basic")
   plt.plot(test_acc, label="Test Basic")
   plt.legend()
   plt.show()
  
