# Written by chatgpt
class EMA:
    def __init__(self, model, decay=0.999):
        """
        Exponential Moving Average (EMA) for model parameters.

        Args:
            model (torch.nn.Module): The model to track.
            decay (float): EMA decay factor, usually close to 1 (e.g., 0.999).
        """
        self.model = model
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self, model):
        """
        Update the EMA parameters with the current model parameters.

        Args:
            model (torch.nn.Module): The current model with updated weights.
        """
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].data.mul_(self.decay).add_((1.0 - self.decay) * param.data)

    def apply_shadow(self):
        """Apply the EMA weights to the model."""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data.copy_(self.shadow[name].data)

    def store(self):
        """Store the current model parameters for restoration."""
        self.backup = {name: param.clone() for name, param in self.model.named_parameters()}

    def restore(self):
        """Restore the original model parameters."""
        for name, param in self.model.named_parameters():
            param.data.copy_(self.backup[name])