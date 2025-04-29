from pytorch_lightning.callbacks import Callback


class MyPrintCallback(Callback):

    def __init__(self):
        super().__init__()

    def on_test_end(self, trainer, pl_module):
        print("Testing finished")
        # Requires implementing
        # fig, ax = pl_module.confusion_matrix.plot()
        # fig.show()  # or savefig if running headless


