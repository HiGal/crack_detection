from sklearn.metrics import precision_recall_fscore_support

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class ClassificationMeter(object):

    def __init__(self):
        pass

    def updatePrecision(self, pred, true):
        pass

    def updateRecall(self, pred, true):
        pass

    def updateF1(self, pred, true):
        pass

    def computePRF1(self, pred, true):
        pred = pred.cpu().numpy()
        true = true.cpu().numpy()
        precision, recall, f1, _ = precision_recall_fscore_support(true, pred, average='micro')
        return precision, recall, f1
