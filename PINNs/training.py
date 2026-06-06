from terminal_condition import terminal_condition
from neural_network import Value_func
from loss_function import loss_function
from collocation_points import collocation_points
import torch
import json

with open(r"C:\Users\Admin\Desktop\HJB_MERTON_BENCHMARK\PINNs\data_accumulation\config.json","r") as file:
    config=json.load(file)

def training_loop(model, optimizer, scheduler, loss_fn, epochs, sigma, rate, mu, x, t, X_t, T_t, gamma):
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = loss_fn(model, x, t, X_t, T_t, gamma, rate, mu, sigma)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if epoch % 1 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")

# sampling collocation_points
t, x = collocation_points(10000, 1, 0.1, 5)
t = t.reshape(-1, 1)
x = x.reshape(-1, 1)

T_t, X_t, V_t = terminal_condition(2000, 1, 5, 0.1, 5)
T_t = T_t.reshape(-1, 1)
X_t = X_t.reshape(-1, 1)

sigma = config["sigma"]
rate = config["rate"]
mu = config["mu"]
gamma = config["gamma"]

model = Value_func()
optimizer = torch.optim.Adam(model.parameters())
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)

training_loop(model, optimizer, scheduler, loss_function, epochs=1000, sigma=sigma, rate=rate, mu=mu, x=x, t=t, X_t=X_t, T_t=T_t, gamma=gamma)