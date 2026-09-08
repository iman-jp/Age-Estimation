# only the training loop is implemented here

def train_one_epoch(model, train_loader, loss_fn, optimizer):
    model.model.train()
    total_loss = 0.0
    num_batches = 0

    for images, ages in train_loader:
        images = images.to("cuda")
        ages = ages.to("cuda").float().unsqueeze(1) 

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