import torch
from logging_utils import log_batch


def validate(model, val_loader, loss_fn, run_id=None, epoch=None, batch_log_path=None):
    model.model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, (images, ages) in enumerate(val_loader):
            images = images.to("cuda")
            ages = ages.to("cuda").float().unsqueeze(1)

            outputs = model.model(images)
            outputs = outputs[1]
            loss = loss_fn(outputs, ages)

            total_loss += loss.item()
            num_batches += 1

            if batch_log_path is not None:
                log_batch(batch_log_path, run_id, epoch, batch_idx, loss.item())

    return total_loss / num_batches