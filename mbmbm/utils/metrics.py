class TorchmetricsClassifAdapter:
    def __init__(self, torchetrics_metrics_cls):
        self.torchetrics_metrics_cls = torchetrics_metrics_cls

    def __call__(self, preds, target, *args, **kwargs):
        return self.torchetrics_metrics_cls(preds, target, *args, **kwargs)
