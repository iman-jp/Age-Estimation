import torch
# only the training loop is implemented here

def train_one_epoch(model, train_loader, loss_fn, optimizer, device="cuda"):
    model.model.train()
    total_loss = 0.0
    num_batches = 0

    for images, ages in train_loader:
        images = images.to(device)
        ages = ages.to(device).float().unsqueeze(1) 

        # 1. forward pass — get predictions
        outputs = model.model(images)
        # 2. compute loss
        loss = loss_fn(outputs, ages)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches

def train_one_epoch_weighted(model, train_loader, optimizer, age_weights, default_weight=1.0, device="cuda"):
    model.model.train()
    total_loss = 0.0
    num_batches = 0

    for images, ages in train_loader:
        images = images.to(device)
        ages_gpu = ages.to(device).float().unsqueeze(1)

        outputs = model.model(images)

        per_sample_loss = torch.abs(outputs - ages_gpu)
        sample_weights = torch.tensor(
            [age_weights.get(int(a.item()), default_weight) for a in ages],
            device=device
        ).unsqueeze(1)

        loss = (per_sample_loss * sample_weights).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / num_batches