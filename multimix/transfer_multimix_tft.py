from multimix.multimix_tft import MultiMixTFT


class MultiMixTFTT(MultiMixTFT):
    """MultiMixTFTT model allows to use a pretrained MultiMixTFT model
        and fine-tune it for a specific task with two outputs.
    Args:
        pretrained_model_path (str): Path to the pretrained MultiMixTFT model checkpoint.

    """

    def __init__(
        self,
        pretrained_model_path: str,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.save_hyperparameters()
        self.pretrained_tft = MultiMixTFT.load_from_checkpoint(pretrained_model_path)
        self.pretrained_tft.freeze()
        self.automatic_optimization = False

        self.build_output_feed_forwards()
        ## Initializing remaining weights
        self.init_weights()
