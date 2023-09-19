'''
returns all predictions,loss,accuracy,number of correct/wrong predictions.
'''
import torch
device = "cuda" if torch.cuda.is_available() else "cpu"

def accuracy_fn(y_true, y_pred):
    """Calculates accuracy between truth labels and predictions.

    Args:
        y_true (torch.Tensor): Truth labels for predictions.
        y_pred (torch.Tensor): Predictions to be compared to truth labels.

    Returns:
        [torch.float]: Accuracy value between y_true and y_pred, e.g. 78.45
    """
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100
    return acc


def get_all_preds(model, dataloader, loss_fn: torch.nn.Module, accuracy_fn=accuracy_fn, device: torch.device = device):
    pred_probs = []
    labels_list = []  # Create a list to store labels
    model.to(device)
    loss, acc, correct, wrong = 0, 0, 0, 0

    model.eval()

    with torch.no_grad():  # Using no_grad context to avoid unnecessary gradient calculations
        for batch in dataloader:
            images, batch_labels = batch  # Use a different variable name for the labels
            images, batch_labels = images.to(device), batch_labels.to(device)

            pred_logit = model(images)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = pred_logit.argmax(dim=1)

            # Get pred_prob off the GPU for further calculations
            pred_probs.append(pred_prob)
            labels_list.append(batch_labels)  # Store the labels in a list

            loss += loss_fn(pred_logit, batch_labels)  # Use batch_labels for loss calculation
            acc += accuracy_fn(y_true=batch_labels, y_pred=pred_prob)
            correct += torch.eq(batch_labels, pred_prob).sum().item()
            wrong += (batch_labels != pred_prob).sum().item()




        # Scale loss and acc to find the average loss/acc per batch
        loss /= len(dataloader)
        acc /= len(dataloader)



    # Convert to item
    pred_probs = torch.cat(pred_probs, dim=0)
    labels = torch.cat(labels_list, dim=0)
    correct += torch.eq(batch_labels, pred_prob).sum().item()
    wrong += (batch_labels != pred_prob).sum().item()  # Concatenate the list of labels into a tensor
    return {
        "pred_probs": pred_probs,
        "labels":labels,
        "model_result":{"model_loss": loss.item(),
                        "model_acc": acc,
                        "correct_predictions": correct,
                        "wrong_predictions": wrong}
          }



