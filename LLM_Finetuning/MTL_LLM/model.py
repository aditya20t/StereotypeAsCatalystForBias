# Import libraries
import torch
from transformers import AutoModel
from peft import LoraConfig, prepare_model_for_kbit_training, get_peft_model

# define MultiTask model
class MultiTaskModel(torch.nn.Module):
        def initialize_weights(self, layer):
            if isinstance(layer, torch.nn.Linear):
                torch.nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)
        
        def __init__(self, model_name, quantization_config, lora_config, class_weights):
            super(MultiTaskModel, self).__init__()
            self.base_model = AutoModel.from_pretrained(model_name, quantization_config=quantization_config)
            print("Base model loaded", self.base_model)

            # Freeze the base model
            for param in self.base_model.parameters():
                param.requires_grad = False

            # Prepare model for kbit training
            self.shared_model = prepare_model_for_kbit_training(self.base_model)
            # Get PEFT model
            self.shared_model = get_peft_model(self.shared_model, lora_config)

            input_size = self.shared_model.config.hidden_size

            self.classifier_bias = torch.nn.Sequential(
                torch.nn.Linear(input_size, input_size//2),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(0.4),
                torch.nn.LayerNorm(input_size//2),
                torch.nn.Linear(input_size//2, input_size//4),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(0.4),
                torch.nn.LayerNorm(input_size//4),
                torch.nn.Linear(input_size//4, 1),
                torch.nn.Sigmoid()
            )

            self.classifier_stereotype = torch.nn.Sequential(
                torch.nn.Linear(input_size, input_size//2),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(0.4),
                torch.nn.LayerNorm(input_size//2),
                torch.nn.Linear(input_size//2, input_size//4),
                torch.nn.LeakyReLU(),
                torch.nn.Dropout(0.4),
                torch.nn.LayerNorm(input_size//4),
                torch.nn.Linear(input_size//4, 1),
                torch.nn.Sigmoid()
            )

            # bias_class_weights = class_weights[:2]
            # stereotype_class_weights = class_weights[2:]

            self.bias_loss_fn = torch.nn.BCELoss()  # Common loss function for classification tasks
            self.stereotype_loss_fn = torch.nn.BCELoss()  # Common loss function for classification tasks

            # Initialize weights for classifiers
            self._initialize_classifier_weights(self.classifier_bias)
            self._initialize_classifier_weights(self.classifier_stereotype)
        
        # Define classifier weights
        def _initialize_classifier_weights(self, classifier):
            for layer in classifier:
                self.initialize_weights(layer)

        # Forward pass
        def forward(self, input_ids=None, attention_mask=None, tasks=None):
            # Pass the input through the shared model to get hidden states
            outputs = self.shared_model(input_ids=input_ids, attention_mask=attention_mask)

            # Choose pooling strategy
            # hidden_states = outputs[0][:, -1, :]  # Take the last hidden state
            hidden_states = torch.mean(outputs[0], dim=1)  # Mean pooling over the hidden states
            # hidden_states = torch.max(outputs[0], dim=1)[0] # Max pooling over the hidden states

            # Initialize lists to hold the logits for each task
            logits_bias_list = []
            logits_stereo_list = []
            
            # Loop through each instance in the batch and compute the logits based on task
            for i in range(input_ids.size(0)):  # Iterate over batch size
                hidden_state = hidden_states[i].unsqueeze(0)  # Get hidden state for the current instance
                if tasks[i] == 0:  # Compute bias logits
                    logits_bias = self.classifier_bias(hidden_state)
                    # print("Multitask model: Logits bias: ", logits_bias.requires_grad)
                    logits_bias_list.append(logits_bias)
                else:  # Compute stereotype logits
                    logits_stereo = self.classifier_stereotype(hidden_state)
                    # print("Multitask model: Logits stereo: ", logits_stereo.requires_grad)
                    logits_stereo_list.append(logits_stereo)
                

            # Stack the logits for bias and stereotype if any logits were computed
            if logits_bias_list:
                logits_bias_stacked = torch.cat(logits_bias_list, dim=0)
            else:
                logits_bias_stacked = None

            if logits_stereo_list:
                logits_stereo_stacked = torch.cat(logits_stereo_list, dim=0)
            else:
                logits_stereo_stacked = None

            # Return the stacked logits
            output = {'logits_bias': logits_bias_stacked, 'logits_stereo': logits_stereo_stacked}
            return output



# define the model function
def get_model(model_name, quantization_config, lora_config, class_weights=None):
     model = MultiTaskModel(model_name, quantization_config, lora_config, class_weights)
     return model