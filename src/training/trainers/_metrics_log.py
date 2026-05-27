"""Per-epoch metrics CSV logger shared by all trainers.

Reads/appends to ``<output_dir>/metrics.csv``. Used by
``plot_training_curves.py`` to render the paper's loss-curves figure.
"""

import csv
import os
import time
from typing import Dict


class MetricsLogger:
    """Appends one row per epoch to a metrics.csv file.

    Header is written on the first row only (or skipped if the file already
    exists, so that resumed runs append cleanly).

    Example:
        logger = MetricsLogger(self.output_dir)
        for epoch in range(num_epochs):
            ...
            logger.log(epoch, lr=opt.param_groups[0]['lr'],
                       metrics={'train/loss': ..., 'val/loss': ...})
    """

    def __init__(self, output_dir: str, filename: str = 'metrics.csv'):
        self.path = os.path.join(output_dir, filename)
        self._need_header = not os.path.exists(self.path)
        self._t0 = time.time()

    def log(self, epoch: int, lr: float, metrics: Dict[str, float]) -> None:
        row = {
            'epoch': epoch,
            'elapsed_sec': round(time.time() - self._t0, 2),
            'lr': lr,
            **metrics,
        }
        with open(self.path, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if self._need_header:
                writer.writeheader()
                self._need_header = False
            writer.writerow(row)
