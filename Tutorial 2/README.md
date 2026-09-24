Task 1
Question: How does increasing the number of layers or neurons affect the accuracy and learning curve?
Answer:Increasing the number of neurons makes training significantly smoother and faster. It gives the model enough representational capacity to separate the XOR inputs easily, causing the loss curve to drop sharply and reach 100% accuracy in fewer epochs. On the other hand, increasing the number of layers (making it deeper) actually hurts training in this setup. Because the network uses standard sigmoid activations without modern weight initialization, extra layers lead to vanishing gradients. This causes the learning curve to flatten out early on and stall, taking far longer to converge.

Question: What configuration gives the best performance?
Answer: The Wider network [2, 8, 1] gives the best performance. It reaches 100% accuracy much faster than the baseline [2, 2, 1] and achieves the lowest final loss without suffering from the vanishing gradient issues that slow down the deeper architectures like [2, 4, 4, 1] and [2, 6, 6, 6, 1


Task 2
Question:Train the model and plot the learning curve.
Answer: The model trains a 2-4-1 MLP on the XOR dataset across 2,500 epochs for each learning rate ($0.01, 0.1, 0.5, 1.0, 5.0, 10.0$) using Binary Cross-Entropy loss. Executing the code generates the convergence curves and final metrics for each run, showing the training loss decreasing from around $0.693$ down toward zero at varying speeds depending on the chosen step size.
Question: How does changing the learning rate affect the loss curve and 
the number of epochs?
Answer: Changing the learning rate directly dictates the slope of the loss curve and the number of epochs required to reach convergence. A very small learning rate like $0.01$ or $0.1$ causes the loss curve to decrease extremely slowly and linearly, remaining stuck near initial loss values and requiring far more than 2,500 epochs to escape saddle points. As the learning rate increases to moderate values like $0.5$ and $1.0$, the curve drops steeply and flattens out in far fewer epochs. However, setting the learning rate excessively high (such as $5.0$ or $10.0$) can cause sharp, aggressive drops early on, but risks visible oscillations, numerical instability, or overshooting optimal weight updates.
Question: What learning rate provides the best balance between 
convergence speed and model performance
Answer: A learning rate of $1.0$ (or $0.5$) provides the best balance between convergence speed and model performance. At this rate, the network reaches near-zero loss and 100% accuracy within roughly 500 to 1,000 epochs without the erratic jumps, saturation, or risk of divergence seen at higher values like $5.0$ and $10.0$, while avoiding the severe underfitting and sluggish progress seen at $0.01$ and $0.1$.
