import torch
import numpy as np

# jakaś własna implementacja sieci
from nnetwork import NeuralNetwork as NNetwork

# jezeli na komputerze i nie ma GPU to piszemy:
# device = torch.device("cpu")

# Jeśli na serwerze to wpisujesz cuda:
# Ta cyfra oznacza na której karcie będziesz uruchamiać skrypt
# Najlepiej wejśc na serwer i wpisać komendę nvidia-smi
# Wtedy zobaczysz która karta jest wolna lub zajęta
device = torch.device("cuda:6")

# Przykład z tensorami
t1 = torch.tensor([1.0, 2.0, 3.0])
print('#1.1', t1.device)
# Zmiana urządzenia
t1.to(device)
print('#1.2', t1.device)

# Najlepiej jest od razu inicjować tensor na odpowiednim urządzeniu
t2 = torch.randint(0, 10, (3, 3), device=device)
print('#2', t2.device)

# Przykład z numpy
a = np.random.rand(3, 3)
t3 = torch.from_numpy(a)
print('#3.1', t3.device)
# Tu niestety nie ma wyjścia i zawsze będzie trzeba zmienić urządzenie
t3.to(device)
print('#3.2', t3.device)

# Przykład z modelem
NNetwork = NNetwork(input_size=3, num_classes=3)
print('#4.1', NNetwork.device)
# Najczęściej trzeba będzie się upewnić że model jest na odpowiednim urządzeniu lub zmienić
NNetwork.to(device)
print('#4.2', NNetwork.device)

# Czasami model może być od razu inicjalizowany na odpowiednim urządzeniu
# To zależy od implementacji
NNetwork = NNetwork(input_size=3, num_classes=3, device=device)
