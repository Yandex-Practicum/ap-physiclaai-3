"""Unit-тест для train_step (Урок 5).

Проверяет, что студент корректно реализовал шаг обучения: считается loss,
обновляются веса, и при повторных шагах на одних данных loss падает.
Тест не зависит от MuJoCo/датасета — использует крошечную модель.
"""
import torch
import torch.nn as nn

from train_bc import train_step


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(4, 8)

    def forward(self, x):
        return self.lin(x)


def test_returns_float_and_updates_weights():
    torch.manual_seed(0)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    obs = torch.randn(16, 4)
    actions = torch.randn(16, 8)

    before = model.lin.weight.detach().clone()
    loss = train_step(model, optimizer, obs, actions)

    assert isinstance(loss, float), "train_step должен возвращать число (loss.item())"
    assert loss >= 0.0
    after = model.lin.weight.detach()
    assert not torch.allclose(before, after), \
        "веса должны обновиться — был ли вызван optimizer.step()?"


def test_loss_decreases_over_steps():
    torch.manual_seed(0)
    model = TinyModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    obs = torch.randn(16, 4)
    actions = torch.randn(16, 8)

    first = train_step(model, optimizer, obs, actions)
    last = first
    for _ in range(30):
        last = train_step(model, optimizer, obs, actions)

    assert last < first, "loss должен падать при повторных шагах на одних данных"
