import mindspore.nn as nn

class ED(nn.Cell):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def construct(self, input):
        state = self.encoder(input)
        output = self.decoder(state)
        return output
