import torch

class FFClassifier(torch.nn.Module):
        def __init__(self, input_size, num_labels, hidden_size=0, dropout=0.0):
            super(FFClassifier, self).__init__()
            self.input_size = input_size
            self.num_labels  = num_labels
            self.hidden_size = hidden_size

            if self.hidden_size > 0:
                self.hidden = torch.nn.Linear(self.input_size, self.hidden_size)
                self.activation = torch.nn.ReLU()
                self.classifier = torch.nn.Linear(self.hidden_size, self.num_labels)
            else:
                self.classifier = torch.nn.Linear(self.input_size, self.num_labels)

            self.dropout = torch.nn.Dropout(dropout)

            self.ce = torch.nn.CrossEntropyLoss()

        def forward(self, x, labels=None):
            if hasattr(self, 'dropout'):
                x = self.dropout(x)

            if hasattr(self, 'hidden_size') and self.hidden_size > 0:
                x = self.activation(self.hidden(x))

            logits = self.classifier(x)
            outputs = (logits,)
            if labels is not None:
                loss = self.ce(logits.view(-1, self.num_labels), labels.view(-1))
                outputs = (loss,) + outputs

            return outputs
