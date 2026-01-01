from micrograd.engine import Value
import math

def mse_loss(preds, targets):
    return sum((ygt - yout)** 2 for ygt, yout in zip(preds, targets)) / len(preds)

def ce_loss(logits, label):
    exps = [math.exp(l.data) for l in logits]
    Z = sum(exps)
    probs = [Value(e/Z) for e in exps]
    loss = -Value(math.log(probs[label].data))
    return loss