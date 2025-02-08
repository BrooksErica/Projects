Neural Networks for Predicting to Detect Particle

Introduction

Our client for this case study is from Case Study 1 with the Superconductors. They’ve come with a much larger data set with less variables than before. In this case study, the goal is to create a dense neural network model that predicts the existence of a new particle with a high level of accuracy.  The detection is binary, with 0 for non-detection and 1 for detection.

Method

The data include over 7,000,000 samples. A binary neural network classifier using PyTorch will be built to make the predictions. The data set contains 28 variables and our target variable for a total of 29 variables. 

Design Decision

The model built for this case study was a PyTorch sequential model using multiple dense layers and three hidden layers. With each layer decreasing in units using powers of 2 (256 -> 128 -> 64 -> 1). Dropout layers were implemented with decreasing rates (0.3 -> 0.1) to prevent overfitting. ReLU activation was applied to the hidden layers to help with the vanishing gradient, and a sigmoid activation was used since this is a binary classification.


Data Pre-Processing

The data file was separated into our data and labels for prediction. The data and labels were split into a train, test, and validation datasets. The training set size had 4,200,000 samples, our validation set size and our test set size both had 1,400,000 samples. The data was then scaled and standardized.

Training Completion

Implemented a custom data class to handle the data. Created a training loop with early stopping monitoring the validation loss. The function also computes the average losses and accuracies for both the training and validation. Training stops when the validation hasn’t improved for 10 Epochs. 

The training loop function uses the following parameters:

| Parameters |	Description |
| ---------- | ----------- |
| Model |	Neural Network model to train |
| Train_loader	Dataloader | containing the training data |
| Val_loader	Dataloader | containing the validation data |
| Criterion |	Loss function: in our case with chose the  Binary Cross Entropy |
| Optimizer |	In our case the Adam optimizer for adaptive learning rates |
| Device |	Run computations on CPU or GPU |
| Num_Epochs |	Number of training Epochs; in our case 100 Epochs |
| Patience |	Number of Epochs to wait before early stopping (10) |


Results

Our early stopping was triggered at 57 Epochs. There was no improvement afterwards. The validation loss was about 88% (0.8841). Model evaluation had a final test accuracy of 88% (0.8839) as well. 

<img width="468" alt="image" src="https://github.com/user-attachments/assets/43f63915-f7e4-407f-94bc-dbfcd00b611c" />



Epoch [1/100], Train Loss: 0.2964, Val Loss: 0.2758, Train Acc: 2.5904, Val Acc: 0.8750  
Epoch [2/100], Train Loss: 0.2795, Val Loss: 0.2698, Train Acc: 2.6192, Val Acc: 0.8783  
Epoch [3/100], Train Loss: 0.2759, Val Loss: 0.2676, Train Acc: 2.6265, Val Acc: 0.8796  
Epoch [4/100], Train Loss: 0.2739, Val Loss: 0.2665, Train Acc: 2.6292, Val Acc: 0.8802  
Epoch [5/100], Train Loss: 0.2729, Val Loss: 0.2658, Train Acc: 2.6310, Val Acc: 0.8806  
Epoch [6/100], Train Loss: 0.2721, Val Loss: 0.2650, Train Acc: 2.6323, Val Acc: 0.8811  
Epoch [7/100], Train Loss: 0.2714, Val Loss: 0.2643, Train Acc: 2.6336, Val Acc: 0.8814  
Epoch [8/100], Train Loss: 0.2709, Val Loss: 0.2645, Train Acc: 2.6347, Val Acc: 0.8810  
Epoch [9/100], Train Loss: 0.2704, Val Loss: 0.2642, Train Acc: 2.6353, Val Acc: 0.8812  
Epoch [10/100], Train Loss: 0.2701, Val Loss: 0.2640, Train Acc: 2.6356, Val Acc: 0.8817  
Epoch [11/100], Train Loss: 0.2699, Val Loss: 0.2633, Train Acc: 2.6363, Val Acc: 0.8819  
Epoch [12/100], Train Loss: 0.2695, Val Loss: 0.2628, Train Acc: 2.6368, Val Acc: 0.8821  
Epoch [13/100], Train Loss: 0.2693, Val Loss: 0.2627, Train Acc: 2.6373, Val Acc: 0.8821  
Epoch [14/100], Train Loss: 0.2691, Val Loss: 0.2634, Train Acc: 2.6374, Val Acc: 0.8820  
Epoch [15/100], Train Loss: 0.2688, Val Loss: 0.2633, Train Acc: 2.6381, Val Acc: 0.8820  
Epoch [16/100], Train Loss: 0.2686, Val Loss: 0.2620, Train Acc: 2.6385, Val Acc: 0.8827  
Epoch [17/100], Train Loss: 0.2685, Val Loss: 0.2619, Train Acc: 2.6390, Val Acc: 0.8826  
Epoch [18/100], Train Loss: 0.2683, Val Loss: 0.2622, Train Acc: 2.6386, Val Acc: 0.8825  
Epoch [19/100], Train Loss: 0.2682, Val Loss: 0.2617, Train Acc: 2.6387, Val Acc: 0.8828  
Epoch [20/100], Train Loss: 0.2680, Val Loss: 0.2623, Train Acc: 2.6390, Val Acc: 0.8827  
Epoch [21/100], Train Loss: 0.2678, Val Loss: 0.2622, Train Acc: 2.6397, Val Acc: 0.8829   
Epoch [22/100], Train Loss: 0.2677, Val Loss: 0.2617, Train Acc: 2.6399, Val Acc: 0.8827  
Epoch [23/100], Train Loss: 0.2677, Val Loss: 0.2614, Train Acc: 2.6396, Val Acc: 0.8829  
Epoch [24/100], Train Loss: 0.2676, Val Loss: 0.2617, Train Acc: 2.6401, Val Acc: 0.8828  
Epoch [25/100], Train Loss: 0.2675, Val Loss: 0.2612, Train Acc: 2.6402, Val Acc: 0.8831  
Epoch [26/100], Train Loss: 0.2673, Val Loss: 0.2616, Train Acc: 2.6402, Val Acc: 0.8832  
Epoch [27/100], Train Loss: 0.2671, Val Loss: 0.2615, Train Acc: 2.6403, Val Acc: 0.8830  
Epoch [28/100], Train Loss: 0.2672, Val Loss: 0.2620, Train Acc: 2.6406, Val Acc: 0.8832  
Epoch [29/100], Train Loss: 0.2669, Val Loss: 0.2611, Train Acc: 2.6410, Val Acc: 0.8831  
Epoch [30/100], Train Loss: 0.2669, Val Loss: 0.2610, Train Acc: 2.6413, Val Acc: 0.8833  
Epoch [31/100], Train Loss: 0.2668, Val Loss: 0.2629, Train Acc: 2.6409, Val Acc: 0.8833  
Epoch [32/100], Train Loss: 0.2667, Val Loss: 0.2613, Train Acc: 2.6417, Val Acc: 0.8833  
Epoch [33/100], Train Loss: 0.2666, Val Loss: 0.2623, Train Acc: 2.6415, Val Acc: 0.8836  
Epoch [34/100], Train Loss: 0.2666, Val Loss: 0.2624, Train Acc: 2.6417, Val Acc: 0.8834  
Epoch [35/100], Train Loss: 0.2665, Val Loss: 0.2613, Train Acc: 2.6419, Val Acc: 0.8834  
Epoch [36/100], Train Loss: 0.2663, Val Loss: 0.2610, Train Acc: 2.6418, Val Acc: 0.8835  
Epoch [37/100], Train Loss: 0.2664, Val Loss: 0.2608, Train Acc: 2.6420, Val Acc: 0.8836  
Epoch [38/100], Train Loss: 0.2662, Val Loss: 0.2604, Train Acc: 2.6423, Val Acc: 0.8836  
Epoch [39/100], Train Loss: 0.2661, Val Loss: 0.2615, Train Acc: 2.6423, Val Acc: 0.8836  
Epoch [40/100], Train Loss: 0.2662, Val Loss: 0.2604, Train Acc: 2.6426, Val Acc: 0.8835  
Epoch [41/100], Train Loss: 0.2662, Val Loss: 0.2610, Train Acc: 2.6422, Val Acc: 0.8837  
Epoch [42/100], Train Loss: 0.2661, Val Loss: 0.2607, Train Acc: 2.6424, Val Acc: 0.8834  
Epoch [43/100], Train Loss: 0.2661, Val Loss: 0.2613, Train Acc: 2.6426, Val Acc: 0.8837  
Epoch [44/100], Train Loss: 0.2659, Val Loss: 0.2618, Train Acc: 2.6428, Val Acc: 0.8837  
Epoch [45/100], Train Loss: 0.2659, Val Loss: 0.2601, Train Acc: 2.6428, Val Acc: 0.8838  
Epoch [46/100], Train Loss: 0.2659, Val Loss: 0.2601, Train Acc: 2.6432, Val Acc: 0.8835  
Epoch [47/100], Train Loss: 0.2659, Val Loss: 0.2601, Train Acc: 2.6429, Val Acc: 0.8839  
Epoch [48/100], Train Loss: 0.2657, Val Loss: 0.2622, Train Acc: 2.6428, Val Acc: 0.8838  
Epoch [49/100], Train Loss: 0.2656, Val Loss: 0.2602, Train Acc: 2.6433, Val Acc: 0.8838  
Epoch [50/100], Train Loss: 0.2658, Val Loss: 0.2611, Train Acc: 2.6429, Val Acc: 0.8838  
Epoch [51/100], Train Loss: 0.2656, Val Loss: 0.2610, Train Acc: 2.6434, Val Acc: 0.8838   
Epoch [52/100], Train Loss: 0.2656, Val Loss: 0.2605, Train Acc: 2.6431, Val Acc: 0.8838  
Epoch [53/100], Train Loss: 0.2656, Val Loss: 0.2604, Train Acc: 2.6433, Val Acc: 0.8840  
Epoch [54/100], Train Loss: 0.2654, Val Loss: 0.2605, Train Acc: 2.6436, Val Acc: 0.8839  
Epoch [55/100], Train Loss: 0.2655, Val Loss: 0.2610, Train Acc: 2.6435, Val Acc: 0.8838  
Epoch [56/100], Train Loss: 0.2656, Val Loss: 0.2606, Train Acc: 2.6431, Val Acc: 0.8841  
Early stopping triggered at epoch 57
