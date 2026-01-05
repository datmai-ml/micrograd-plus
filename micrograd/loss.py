from micrograd.engine import Value
import math

def mse_loss(preds, targets):
    return sum((ygt - yout)**2 for ygt, yout in zip(preds, targets)) / len(preds)

def softmax(logits):
    exps = [l.exp() for l in logits]
    Z = sum(exps)
    probs = [e / Z for e in exps]
    return probs

def ce_loss(logits_batch, labels):
    total_loss = Value(0.0)
    for logits, label in zip(logits_batch, labels):
        probs = softmax(logits)
        log_prob = probs[label].log()
        total_loss += -log_prob
    return total_loss / len(labels)